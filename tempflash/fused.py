# TempFlash block-free execution: the sampler's (row, slot) layout is consumed
# directly, so no DGL graph object is built per batch. The attention layer below
# is the same computation as TGL's TransfomerAttentionLayer (non-combined path):
# identical parameters and linear maps; edge softmax over a root's sampled
# neighbours becomes a masked softmax over its K slots, and the copy_u/sum
# aggregation becomes a dense slot sum (deterministic, no atomics).
import torch
import layers as _layers

_orig_forward = _layers.TransfomerAttentionLayer.forward


class FlatBlock:
    """dst = R roots; src = [R roots, M sampled neighbours]; edge e: src R+e -> dst row[e]."""
    __slots__ = ('R', 'M', 'K', 'row', 'slot', 'srcdata', 'edata', 'dstdata')

    def __init__(self, R, M, K, row, slot):
        self.R, self.M, self.K, self.row, self.slot = R, M, K, row, slot
        self.srcdata, self.edata, self.dstdata = {}, {}, {}

    def num_src_nodes(self):
        return self.R + self.M

    def num_dst_nodes(self):
        return self.R

    def num_edges(self):
        return self.M


class DenseBlock:
    """Static-shape block: R roots, each with K neighbour slots and a validity mask.
    srcdata rows = [R roots, R*K slots (row-major)]; edata rows = R*K slots."""
    __slots__ = ('R', 'K', 'valid', 'has_nb', 'srcdata', 'edata', 'dstdata', 'ctx')

    def __init__(self, R, K, valid, has_nb=True):
        self.R, self.K, self.valid, self.has_nb = R, K, valid, has_nb
        self.srcdata, self.edata, self.dstdata = {}, {}, {}
        self.ctx = None

    def num_src_nodes(self):
        return self.R * (self.K + 1) if self.has_nb else self.R

    def num_dst_nodes(self):
        return self.R

    def num_edges(self):
        return self.R * self.K if self.has_nb else 0


def dense_forward(self, b):
    """TGL TransfomerAttentionLayer on a (R, K) slot layout; masked slots carry zero
    attention, a batch with no valid neighbour returns zeros exactly like the DGL path."""
    R, Kn = b.R, b.K
    dev = b.valid.device
    valid = b.valid
    if self.dim_node_feat > 0:
        h = b.srcdata['h']
        h_root, h_nb = h[:R], h[R:]
    if self.dim_time > 0:
        time_feat = self.time_enc(b.edata['dt'])
        zero_time_feat = self.time_enc(torch.zeros(R, dtype=torch.float32, device=dev))
    if self.dim_time == 0 and self.dim_node_feat == 0:
        Q = torch.ones((R, self.dim_out), device=dev)
        kv = b.edata['f']
    elif self.dim_time == 0 and self.dim_edge_feat == 0:
        Q = self.w_q(h_root)
        kv = h_nb
    elif self.dim_time == 0:
        Q = self.w_q(h_root)
        kv = torch.cat([h_nb, b.edata['f']], dim=1)
    elif self.dim_node_feat == 0:
        Q = self.w_q(zero_time_feat)
        kv = torch.cat([b.edata['f'], time_feat], dim=1)
    elif self.dim_edge_feat == 0:
        Q = self.w_q(torch.cat([h_root, zero_time_feat], dim=1))
        kv = torch.cat([h_nb, time_feat], dim=1)
    else:
        Q = self.w_q(torch.cat([h_root, zero_time_feat], dim=1))
        kv = torch.cat([h_nb, b.edata['f'], time_feat], dim=1)
    H = self.num_head
    K = self.w_k(kv).reshape(R, Kn, H, -1)
    V = self.w_v(kv).reshape(R, Kn, H, -1)
    Q = Q.reshape(R, 1, H, -1)
    logits = self.att_act(torch.sum(Q * K, dim=3))                       # (R, K, H)
    vm = valid.unsqueeze(2)
    logits = logits.masked_fill(~vm, torch.finfo(logits.dtype).min)
    att = torch.softmax(logits, dim=1) * vm
    att = self.att_dropout(att)
    agg = (V * att.unsqueeze(3)).sum(dim=1).reshape(R, -1)
    if self.dim_node_feat != 0:
        rst = torch.cat([agg, h_root], dim=1)
    else:
        rst = agg
    rst = self.w_out(rst)
    rst = torch.nn.functional.relu(self.dropout(rst))
    rst = self.layer_norm(rst)
    return torch.where(valid.any(), rst, torch.zeros_like(rst))


def fused_forward(self, b):
    if isinstance(b, DenseBlock):
        assert not self.combined
        return dense_forward(self, b)
    if not isinstance(b, FlatBlock):
        return _orig_forward(self, b)
    assert not self.combined, 'combined attention is not used by the ETC configs'
    R, M = b.R, b.M
    dev = b.row.device
    if M == 0:
        return torch.zeros((R, self.dim_out), device=dev)
    row = b.row
    if self.dim_node_feat > 0:
        h = b.srcdata['h']
        h_root, h_nb = h[:R], h[R:]
    if self.dim_time > 0:
        time_feat = self.time_enc(b.edata['dt'])
        zero_time_feat = self.time_enc(torch.zeros(R, dtype=torch.float32, device=dev))
    if self.dim_time == 0 and self.dim_node_feat == 0:
        Q = torch.ones((M, self.dim_out), device=dev)
        K = self.w_k(b.edata['f'])
        V = self.w_v(b.edata['f'])
    elif self.dim_time == 0 and self.dim_edge_feat == 0:
        Q = self.w_q(h_root)[row]
        K = self.w_k(h_nb)
        V = self.w_v(h_nb)
    elif self.dim_time == 0:
        Q = self.w_q(h_root)[row]
        kv = torch.cat([h_nb, b.edata['f']], dim=1)
        K, V = self.w_k(kv), self.w_v(kv)
    elif self.dim_node_feat == 0:
        Q = self.w_q(zero_time_feat)[row]
        kv = torch.cat([b.edata['f'], time_feat], dim=1)
        K, V = self.w_k(kv), self.w_v(kv)
    elif self.dim_edge_feat == 0:
        Q = self.w_q(torch.cat([h_root, zero_time_feat], dim=1))[row]
        kv = torch.cat([h_nb, time_feat], dim=1)
        K, V = self.w_k(kv), self.w_v(kv)
    else:
        Q = self.w_q(torch.cat([h_root, zero_time_feat], dim=1))[row]
        kv = torch.cat([h_nb, b.edata['f'], time_feat], dim=1)
        K, V = self.w_k(kv), self.w_v(kv)
    H = self.num_head
    Q = Q.reshape(M, H, -1)
    K = K.reshape(M, H, -1)
    V = V.reshape(M, H, -1)
    logits = self.att_act(torch.sum(Q * K, dim=2))                       # (M, H)
    dense = torch.full((R, b.K, H), torch.finfo(logits.dtype).min, device=dev, dtype=logits.dtype)
    dense = dense.index_put((row, b.slot), logits)
    att = torch.softmax(dense, dim=1)[row, b.slot]                        # (M, H)
    att = self.att_dropout(att)
    Vw = (V * att[:, :, None]).reshape(M, -1)
    agg = torch.zeros((R, b.K, Vw.shape[1]), device=dev, dtype=Vw.dtype)
    agg = agg.index_put((row, b.slot), Vw).sum(dim=1)                     # (R, dim_out)
    if self.dim_node_feat != 0:
        rst = torch.cat([agg, h_root], dim=1)
    else:
        rst = agg
    rst = self.w_out(rst)
    rst = torch.nn.functional.relu(self.dropout(rst))
    return self.layer_norm(rst)


def install():
    _layers.TransfomerAttentionLayer.forward = fused_forward
