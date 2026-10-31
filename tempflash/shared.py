# TempFlash projection reordering: "project once, gather many".
#
# Exact algebraic rewrites of the TGL linear maps on TempFlash's static slot layout:
#   (1) column split -- W [x_const ; x_grad] + b = W_c x_const + W_g x_grad + b.
#       Constant inputs (raw node/edge features, stored memory and mails) never
#       receive a gradient, and the concatenated input is never materialised;
#       K and V share one GEMM per input block.
#   (2) node sharing -- a term that depends only on the node id (raw node feature,
#       stored memory, stored mail) is projected once per graph node and gathered
#       per row whenever the graph has no more nodes than the step has rows
#       (decided per size bucket at capture time, so it is free at run time).
# Parameters, initialisation and the computed function are exactly TGL's; only the
# evaluation order of the linear maps changes (fp32 rounding differences only).
import torch
import torch.nn.functional as F
import layers as _layers
import memorys as _memorys
import fused

ENABLED = True
STATS = {}
_orig_gru = _memorys.GRUMemeoryUpdater.forward
_orig_tmu = _memorys.TransformerMemoryUpdater.forward


class Ctx:
    """Per-engine handles used by the reordered modules (tables stay resident)."""

    def __init__(self, N, node_t, edge_t, st, share=True):
        self.N, self.node_t, self.edge_t, self.st, self.share = int(N), node_t, edge_t, st, bool(share)

    def node_rows(self, idx):
        return self.node_t.gather(idx)

    def node_all(self):
        return self.node_t.all()

    def can_share(self, rows, node_table=True):
        if not self.share or self.N > rows:
            return False
        return (not node_table) or (self.node_t is not None and hasattr(self.node_t, 'all'))


def _note(key, val):
    STATS[key] = val


def mail_blocks(st, nid, edge_t):
    """Stored mails of `nid` as column blocks [(tensor (n,S,c), col0)], never concatenated.
    Factorized mailboxes re-gather the edge-feature block by event id."""
    m = st.mail.index_select(0, nid)
    blocks = [(m, 0)]
    if st.factorize:
        e = st.mail_eid.index_select(0, nid)
        ef = edge_t.gather(e.clamp(min=0).reshape(-1)).reshape(e.shape[0], e.shape[1], -1)
        ef = torch.where((e >= 0).unsqueeze(2), ef, torch.zeros_like(ef))
        blocks.append((ef, m.shape[2]))
    return blocks


def proj_blocks(blocks, W):
    out = None
    for x, c0 in blocks:
        y = F.linear(x, W[:, c0:c0 + x.shape[-1]])
        out = y if out is None else out + y
    return out


# --------------------------------------------------------------- attention
def att_forward(self, b):
    ctx = getattr(b, 'ctx', None) if isinstance(b, fused.DenseBlock) else None
    if not ENABLED or ctx is None or self.combined:
        return fused.fused_forward(self, b)
    R, Kn = b.R, b.K
    valid = b.valid
    dev = valid.device
    dn, de, dt, d = self.dim_node_feat, self.dim_edge_feat, self.dim_time, self.dim_out
    H = self.num_head
    ids = b.srcdata['ID']
    h = b.srcdata.get('h', None)                       # per-row h from a memory updater
    if dn > 0:
        h_root = h[:R] if h is not None else ctx.node_rows(ids[:R])
    if dt > 0:
        zero_time_feat = self.time_enc(torch.zeros(R, dtype=torch.float32, device=dev))
    # query (roots only; small)
    if dt == 0 and dn == 0:
        Q = torch.ones((R, d), device=dev)
    elif dt == 0:
        Q = self.w_q(h_root)
    elif dn == 0:
        Q = self.w_q(zero_time_feat)
    else:
        Q = self.w_q(torch.cat([h_root, zero_time_feat], dim=1))
    # keys / values: one GEMM per input block, K and V stacked
    W = torch.cat([self.w_k.weight, self.w_v.weight], 0)
    KV = None
    c0 = 0
    if dn > 0:
        Wh = W[:, :dn]
        c0 = dn
        if h is not None:
            KV = F.linear(h[R:], Wh)
        elif ctx.can_share(R * Kn):
            KV = F.linear(ctx.node_all(), Wh).index_select(0, ids[R:])
            _note('att_shared', True)
        else:
            KV = F.linear(ctx.node_rows(ids[R:]), Wh)
    if de > 0:
        y = F.linear(b.edata['f'], W[:, c0:c0 + de])
        c0 += de
        KV = y if KV is None else KV + y
    if dt > 0:
        y = F.linear(self.time_enc(b.edata['dt']), W[:, c0:c0 + dt])
        KV = y if KV is None else KV + y
    KV = KV + torch.cat([self.w_k.bias, self.w_v.bias], 0)
    Kt = KV[:, :d].reshape(R, Kn, H, -1)
    Vt = KV[:, d:].reshape(R, Kn, H, -1)
    Qt = Q.reshape(R, 1, H, -1)
    logits = self.att_act(torch.sum(Qt * Kt, dim=3))
    vm = valid.unsqueeze(2)
    logits = logits.masked_fill(~vm, torch.finfo(logits.dtype).min)
    att = torch.softmax(logits, dim=1) * vm
    att = self.att_dropout(att)
    agg = (Vt * att.unsqueeze(3)).sum(dim=1).reshape(R, -1)
    rst = torch.cat([agg, h_root], dim=1) if dn != 0 else agg
    rst = self.w_out(rst)
    rst = torch.nn.functional.relu(self.dropout(rst))
    rst = self.layer_norm(rst)
    return torch.where(valid.any(), rst, torch.zeros_like(rst))


# --------------------------------------------------------------- TGN / JODIE memory (GRU)
def gru_forward(self, mfg):
    for b in mfg:
        ctx = getattr(b, 'ctx', None)
        if not ENABLED or ctx is None:
            return _orig_gru(self, [b])
        st = ctx.st
        nid = b.srcdata['ID']
        ts = b.srcdata['ts']
        rows = nid.numel()
        mem_ts = st.node_memory_ts.index_select(0, nid)
        hx = st.node_memory.index_select(0, nid)
        cell = self.updater
        Wih, Whh = cell.weight_ih, cell.weight_hh
        dmail = Wih.shape[1] - self.dim_time
        if ctx.can_share(rows, node_table=False):
            allid = torch.arange(ctx.N, device=nid.device)
            blocks = [(x.reshape(ctx.N, -1), c) for x, c in mail_blocks(st, allid, ctx.edge_t)]
            gi = proj_blocks(blocks, Wih[:, :dmail]).index_select(0, nid)
            gh = F.linear(st.node_memory[:ctx.N], Whh).index_select(0, nid)
            _note('gru_shared', True)
        else:
            blocks = [(x.reshape(rows, -1), c) for x, c in mail_blocks(st, nid, ctx.edge_t)]
            gi = proj_blocks(blocks, Wih[:, :dmail])
            gh = F.linear(hx, Whh)
        if self.dim_time > 0:
            gi = gi + F.linear(self.time_enc(ts - mem_ts), Wih[:, dmail:])
        updated_memory = torch.ops.aten._thnn_fused_gru_cell(gi, gh, hx, cell.bias_ih, cell.bias_hh)[0]
        self.last_updated_ts = ts.detach().clone()
        self.last_updated_memory = updated_memory.detach().clone()
        self.last_updated_nid = nid.detach().clone()
        b.srcdata['mem_ts'] = mem_ts
        if self.memory_param['combine_node_feature'] and self.dim_node_feat > 0:
            if self.dim_node_feat == self.dim_hid:
                b.srcdata['h'] = ctx.node_rows(nid) + updated_memory
            elif ctx.can_share(rows):
                b.srcdata['h'] = updated_memory + self.node_feat_map(ctx.node_all()).index_select(0, nid)
                _note('nodemap_shared', True)
            else:
                b.srcdata['h'] = updated_memory + self.node_feat_map(ctx.node_rows(nid))
        else:
            b.srcdata['h'] = updated_memory


# --------------------------------------------------------------- APAN memory (mailbox attention)
def tmu_forward(self, mfg):
    for b in mfg:
        ctx = getattr(b, 'ctx', None)
        if not ENABLED or ctx is None:
            return _orig_tmu(self, [b])
        st = ctx.st
        nid = b.srcdata['ID']
        ts = b.srcdata['ts']
        R = nid.numel()
        S = self.memory_param['mailbox_size']
        mem = st.node_memory.index_select(0, nid)
        Q = self.w_q(mem).reshape((R, self.att_h, -1))
        W = torch.cat([self.w_k.weight, self.w_v.weight], 0)
        dmail = W.shape[1] - self.dim_time
        if ctx.can_share(R, node_table=False):
            allid = torch.arange(ctx.N, device=nid.device)
            KV = proj_blocks(mail_blocks(st, allid, ctx.edge_t), W[:, :dmail]).index_select(0, nid)
            _note('tmu_shared', True)
        else:
            KV = proj_blocks(mail_blocks(st, nid, ctx.edge_t), W[:, :dmail])
        if self.dim_time > 0:
            mail_ts = st.mailbox_ts.index_select(0, nid)
            time_feat = self.time_enc(ts[:, None] - mail_ts).reshape((R, S, -1))
            KV = KV + F.linear(time_feat, W[:, dmail:])
        KV = KV + torch.cat([self.w_k.bias, self.w_v.bias], 0)
        d = self.w_k.weight.shape[0]
        K = KV[..., :d].reshape((R, S, self.att_h, -1))
        V = KV[..., d:].reshape((R, S, self.att_h, -1))
        att = self.att_act((Q[:, None, :, :] * K).sum(dim=3))
        att = torch.nn.functional.softmax(att, dim=1)
        att = self.att_dropout(att)
        rst = (att[:, :, :, None] * V).sum(dim=1)
        rst = rst.reshape((rst.shape[0], -1))
        rst = rst + mem
        rst = self.layer_norm(rst)
        rst = self.mlp(rst)
        rst = self.dropout(rst)
        rst = torch.nn.functional.relu(rst)
        b.srcdata['h'] = rst
        self.last_updated_memory = rst.detach().clone()
        self.last_updated_nid = nid.detach().clone()
        self.last_updated_ts = ts.detach().clone()


def install():
    fused.install()
    _layers.TransfomerAttentionLayer.forward = att_forward
    _memorys.GRUMemeoryUpdater.forward = gru_forward
    _memorys.TransformerMemoryUpdater.forward = tmu_forward
