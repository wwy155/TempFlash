# TempFlash engine: one T-GNN training/eval step as a static-shape, CPU-free GPU
# program that can be captured once as a CUDA graph and replayed per batch.
#
#   batch slice (device cursor) -> negative draw -> exact recent sampling
#   -> slot-layout gather of features / memory / mailbox -> forward/backward/Adam
#   -> deterministic last-writer memory + mailbox merge
#
# Batches are padded to a small set of size buckets (masked rows never touch
# loss, sampler pointers, memory or mailbox); one graph per bucket and phase,
# all sharing one memory pool. Semantics follow TGL/ETC exactly; intra-batch
# write conflicts are resolved by event order (last writer wins).
import numpy as np
import torch
import torch.nn.functional as F
import fused

GB = 2 ** 30


def round_up(x, m):
    return int((int(np.ceil(x)) + m - 1) // m * m)


def make_buckets(sizes, nbuckets=4, mult=64):
    sizes = np.asarray(sizes)
    mx = int(sizes.max())
    if nbuckets <= 1:
        return [mx]
    qs = np.linspace(0.5, 0.95, nbuckets - 1)
    b = sorted(set([min(round_up(np.quantile(sizes, q), mult), mx) for q in qs] + [mx]))
    return b


class StateTables:
    """Node memory + mailbox, resident on the GPU, with one trash row (index N).
    factorize=True stores each mail as [own memory, other memory] plus the id of
    the event that produced it; the immutable edge-feature part is re-gathered by
    id when the mailbox is read (lossless, saves dim_edge floats per slot)."""

    def __init__(self, mp, N, dim_edge, dev, factorize=False):
        self.mp, self.N, self.dev = mp, N, dev
        self.S = mp['mailbox_size']
        self.dm = mp['dim_out']
        self.de = dim_edge
        self.factorize = bool(factorize and dim_edge > 0)
        dst = 2 * self.dm if self.factorize else 2 * self.dm + dim_edge
        self.node_memory = torch.zeros((N + 1, self.dm), device=dev)
        self.node_memory_ts = torch.zeros(N + 1, device=dev)
        self.mail = torch.zeros((N + 1, self.S, dst), device=dev)
        self.mail_eid = torch.full((N + 1, self.S), -1, dtype=torch.int64, device=dev) if self.factorize else None
        self.mailbox_ts = torch.zeros((N + 1, self.S), device=dev)
        self.next_pos = torch.zeros(N + 1, dtype=torch.int64, device=dev)
        self.win = torch.full((N + 1,), -1, dtype=torch.int64, device=dev)

    @staticmethod
    def full_bytes(mp, N, dim_edge):
        return (N + 1) * mp['mailbox_size'] * (2 * mp['dim_out'] + dim_edge) * 4

    def nbytes(self):
        ts = [self.node_memory, self.node_memory_ts, self.mail, self.mailbox_ts, self.next_pos, self.win]
        if self.mail_eid is not None:
            ts.append(self.mail_eid)
        return sum(t.numel() * t.element_size() for t in ts)

    def reset(self):
        for t in (self.node_memory, self.node_memory_ts, self.mail, self.mailbox_ts, self.next_pos):
            t.zero_()
        if self.mail_eid is not None:
            self.mail_eid.fill_(-1)
        self.win.fill_(-1)

    def winners(self, nid, valid):
        """dest[i] = nid[i] if i is the last valid occurrence of nid[i], else the trash row."""
        pos = torch.arange(nid.numel(), device=nid.device)
        tgt = torch.where(valid, nid, torch.full_like(nid, self.N))
        self.win.scatter_reduce_(0, tgt, torch.where(valid, pos, torch.full_like(pos, -1)),
                                 reduce='amax', include_self=True)
        is_win = (self.win.index_select(0, nid) == pos) & valid
        self.win.index_fill_(0, tgt, -1)
        return torch.where(is_win, nid, torch.full_like(nid, self.N))

    def attach(self, b, nid, edge_t):
        R = nid.numel()
        b.srcdata['mem'] = self.node_memory.index_select(0, nid)
        b.srcdata['mem_ts'] = self.node_memory_ts.index_select(0, nid)
        m = self.mail.index_select(0, nid)
        if self.factorize:
            e = self.mail_eid.index_select(0, nid)
            ef = edge_t.gather(e.clamp(min=0).reshape(-1)).reshape(R, self.S, -1)
            ef = torch.where((e >= 0).unsqueeze(2), ef, torch.zeros_like(ef))
            m = torch.cat([m, ef], dim=2)
        b.srcdata['mem_input'] = m.reshape(R, -1)
        b.srcdata['mail_ts'] = self.mailbox_ts.index_select(0, nid)

    def write_mail(self, dest, mail, mail_ts, mail_eid):
        p = self.next_pos.index_select(0, dest)
        self.mail.index_put_((dest, p), mail)
        if self.factorize:
            self.mail_eid.index_put_((dest, p), mail_eid)
        self.mailbox_ts.index_put_((dest, p), mail_ts)
        if self.S > 1:
            self.next_pos.index_put_((dest,), torch.remainder(p + 1, self.S))


def masked_loss(pp, pn, ev_valid, n):
    w = ev_valid.to(pp.dtype)
    lp = F.binary_cross_entropy_with_logits(pp.squeeze(1), torch.ones_like(w), reduction='none')
    ln = F.binary_cross_entropy_with_logits(pn.squeeze(1), torch.zeros_like(w), reduction='none')
    cnt = n.to(pp.dtype)
    return (lp * w).sum() / cnt + (ln * w).sum() / cnt


@torch.no_grad()
def batched_ap(store, sizes_np, nb_max, chunk=256):
    """Per-batch average precision on the GPU, same definition as sklearn's
    average_precision_score on sigmoid scores: AP = sum over distinct-score groups
    (descending) of (delta recall) * precision at the group end."""
    dev = store.device
    sizes = torch.as_tensor(np.asarray(sizes_np), device=dev)
    B = len(sizes_np)
    out = torch.empty(B, dtype=torch.float64, device=dev)
    ar = torch.arange(nb_max, device=dev)
    for c0 in range(0, B, chunk):
        c1 = min(B, c0 + chunk)
        m = c1 - c0
        s = store[c0 * nb_max:c1 * nb_max].reshape(m, nb_max, 2)
        n = sizes[c0:c1]
        v = ar.unsqueeze(0) < n.unsqueeze(1)
        sc = torch.cat([s[:, :, 0], s[:, :, 1]], dim=1).sigmoid()
        lab = torch.cat([torch.ones(m, nb_max, device=dev), torch.zeros(m, nb_max, device=dev)], dim=1)
        vv = torch.cat([v, v], dim=1)
        sc = torch.where(vv, sc, torch.full_like(sc, -1.0))          # padded entries sort last
        ss, idx = torch.sort(sc, dim=1, descending=True, stable=True)
        lab = torch.gather(lab, 1, idx)
        vv = torch.gather(vv, 1, idx)
        tp = torch.cumsum(lab, dim=1).double()
        k = torch.arange(1, 2 * nb_max + 1, device=dev, dtype=torch.float64).unsqueeze(0)
        prec = tp / k
        nxt_s = torch.cat([ss[:, 1:], torch.full((m, 1), -2.0, device=dev)], dim=1)
        nxt_v = torch.cat([vv[:, 1:], torch.zeros((m, 1), dtype=torch.bool, device=dev)], dim=1)
        is_end = vv & ((ss != nxt_s) | ~nxt_v)
        pos = torch.arange(2 * nb_max, device=dev).unsqueeze(0).expand(m, -1)
        e_idx = torch.where(is_end, pos, torch.full_like(pos, 2 * nb_max - 1))
        eog = torch.flip(torch.cummin(torch.flip(e_idx, [1]), dim=1).values, [1])
        contrib = torch.gather(prec, 1, eog) * (lab.double() * vv.double())
        out[c0:c1] = contrib.sum(1) / n.double()
    return out


class Engine:
    def __init__(self, model, optimizer, sampler, node_t, edge_t, events, cfg, st, buckets,
                 n_slots, eval_cap, dev, use_node_feats=True, external_neg=False, reorder=False):
        sp, mp, gp, tp = cfg
        self.model, self.opt, self.sampler = model, optimizer, sampler
        self.node_t = node_t if use_node_feats else None
        self.edge_t, self.st = edge_t, st
        self.gSRC, self.gDST, self.gTIME, self.gEID = events
        self.E = self.gSRC.numel()
        self.N = sampler.num_nodes
        self.K = sampler.k
        self.batch_size = tp['batch_size']
        self.identity = gp['arch'] == 'identity'
        self.no_neg = bool(sp.get('no_neg', False))
        self.deliver_nb = st is not None and mp['deliver_to'] == 'neighbors'
        self.dev = dev
        self.buckets = list(buckets)
        self.NBmax = self.buckets[-1]
        self.external_neg = external_neg
        self.starts = torch.zeros(n_slots, dtype=torch.int64, device=dev)
        self.sizes = torch.zeros(n_slots, dtype=torch.int64, device=dev)
        self.bidx = torch.zeros(1, dtype=torch.int64, device=dev)
        self.tot_loss = torch.zeros((), device=dev)
        self.neg_buf = torch.zeros(self.NBmax, dtype=torch.int64, device=dev)
        self.eval_cap = int(eval_cap)
        self.pred_store = torch.zeros((self.eval_cap * self.NBmax, 2), device=dev)
        self.ar = {nb: torch.arange(nb, device=dev) for nb in self.buckets}
        self.ar2 = {nb: torch.arange(2 * nb, device=dev) for nb in self.buckets}
        self.g_train, self.g_eval = {}, {}
        self.pool = None
        self.host_sizes = None
        self.host_bucket = None
        # projection reordering (shared.py): modules gather their own inputs
        import shared
        self.reorder = bool(reorder)
        self.ctx = shared.Ctx(self.N, self.node_t, edge_t, st)

    # ------------------------------------------------------------------ phases
    def load_phase(self, rng):
        r = np.asarray(rng, dtype=np.int64)
        self.starts.zero_()
        self.sizes.zero_()
        self.starts[:len(r)].copy_(torch.from_numpy(r[:, 0]).to(self.dev))
        self.sizes[:len(r)].copy_(torch.from_numpy(r[:, 1] - r[:, 0]).to(self.dev))
        self.bidx.zero_()
        self.host_sizes = (r[:, 1] - r[:, 0])
        bk = np.asarray(self.buckets)
        self.host_bucket = bk[np.searchsorted(bk, self.host_sizes, side='left')]

    def reset_state(self):
        self.sampler.reset()
        if self.st is not None:
            self.st.reset()
        self.bidx.zero_()
        self.tot_loss.zero_()

    # ------------------------------------------------------------------ step
    def assemble(self, nb):
        AR = self.ar[nb]
        lo = self.starts.index_select(0, self.bidx).view(())
        n = self.sizes.index_select(0, self.bidx).view(())
        ev_valid = AR < n
        ev = torch.clamp(lo + AR, max=self.E - 1)
        src = self.gSRC.index_select(0, ev).long()
        dst = self.gDST.index_select(0, ev).long()
        t = self.gTIME.index_select(0, ev)
        eid = ev if self.gEID is None else self.gEID.index_select(0, ev).long()
        if self.external_neg:
            neg = self.neg_buf[:nb]
        else:
            neg = torch.randint(0, self.N, (nb,), device=self.dev)
        root = torch.cat([src, dst, neg])
        ts = torch.cat([t, t, t])
        rvalid = torch.cat([ev_valid, ev_valid, ev_valid])
        bt = dict(nb=nb, n=n, ev_valid=ev_valid, src=src, dst=dst, t=t, eid=eid, root=root, ts=ts, rvalid=rvalid)
        if self.identity:
            b = fused.DenseBlock(3 * nb, self.K, None, has_nb=False)
            b.srcdata['ID'] = root
            b.srcdata['ts'] = ts
            nid = root
            if self.deliver_nb:
                pr = 2 * nb if self.no_neg else 3 * nb
                bt['pr'] = pr
                bt['nbr'] = self.sampler.sample_dense(root[:pr], ts[:pr], rvalid[:pr])
        else:
            if self.no_neg:
                raise NotImplementedError('no_neg with message passing is not used by the ETC configs')
            valid, nb_node, nb_eid, nb_ts, nb_dt = self.sampler.sample_dense(root, ts, rvalid)
            b = fused.DenseBlock(3 * nb, self.K, valid, has_nb=True)
            nid = torch.cat([root, nb_node.reshape(-1)])
            b.srcdata['ID'] = nid
            b.srcdata['ts'] = torch.cat([ts, nb_ts.reshape(-1)])
            b.edata['dt'] = nb_dt.reshape(-1)
            if self.edge_t is not None:
                b.edata['f'] = self.edge_t.gather(nb_eid.reshape(-1))
        if self.reorder:
            b.ctx = self.ctx
        else:
            if self.node_t is not None:
                b.srcdata['h'] = self.node_t.gather(nid)
            if self.st is not None:
                self.st.attach(b, nid, self.edge_t)
        return b, bt

    @torch.no_grad()
    def writeback(self, bt):
        st = self.st
        mu = self.model.memory_updater
        nb, n = bt['nb'], bt['n']
        upd = mu.last_updated_memory
        mem_src, mem_dst = upd[:nb], upd[nb:2 * nb]
        src_mail = torch.cat([mem_src, mem_dst], dim=1)
        dst_mail = torch.cat([mem_dst, mem_src], dim=1)
        if self.edge_t is not None and not st.factorize:
            ef = self.edge_t.gather(bt['eid'])
            src_mail = torch.cat([src_mail, ef], dim=1)
            dst_mail = torch.cat([dst_mail, ef], dim=1)
        if not self.deliver_nb:
            # TGN/JODIE: interleaved (src_i, dst_i) mails; TGL pairs them with ts[:2n]
            AR2 = self.ar2[nb]
            mail = torch.stack([src_mail, dst_mail], dim=1).reshape(2 * nb, -1)
            nid = torch.stack([bt['src'], bt['dst']], dim=1).reshape(-1)
            meid = torch.stack([bt['eid'], bt['eid']], dim=1).reshape(-1)
            idx = torch.where(AR2 < n, AR2, AR2 - n).clamp(0, nb - 1)
            mail_ts = bt['t'].index_select(0, idx)
            dest = st.winners(nid, AR2 < 2 * n)
            st.write_mail(dest, mail, mail_ts, meid)
        else:
            # APAN: each root's mail goes to the root and to its sampled neighbours
            valid, nb_node = bt['nbr'][0], bt['nbr'][1]
            pr = bt['pr']
            if pr != 2 * nb:
                raise NotImplementedError
            K = self.K
            mail_r = torch.cat([src_mail, dst_mail], dim=0)
            ts_r = torch.cat([bt['t'], bt['t']])
            eid_r = torch.cat([bt['eid'], bt['eid']])
            seq_nid = torch.cat([bt['root'][:pr], nb_node.reshape(-1)])
            seq_mail = torch.cat([mail_r, mail_r.unsqueeze(1).expand(pr, K, mail_r.shape[1]).reshape(pr * K, -1)])
            seq_ts = torch.cat([ts_r, ts_r.unsqueeze(1).expand(pr, K).reshape(-1)])
            seq_eid = torch.cat([eid_r, eid_r.unsqueeze(1).expand(pr, K).reshape(-1)])
            seq_valid = torch.cat([bt['rvalid'][:pr], valid.reshape(-1)])
            dest = st.winners(seq_nid, seq_valid)
            st.write_mail(dest, seq_mail, seq_ts, seq_eid)
        nid = mu.last_updated_nid[:2 * nb]
        dest = st.winners(nid, bt['rvalid'][:2 * nb])
        st.node_memory.index_put_((dest,), mu.last_updated_memory[:2 * nb])
        st.node_memory_ts.index_put_((dest,), mu.last_updated_ts[:2 * nb])

    def train_body(self, nb):
        b, bt = self.assemble(nb)
        pp, pn = self.model([[b]])
        loss = masked_loss(pp, pn, bt['ev_valid'], bt['n'])
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.tot_loss.add_(loss.detach() * self.batch_size)
            if self.st is not None:
                self.writeback(bt)
            self.bidx.add_(1)
        return pp, pn

    def eval_body(self, nb):
        with torch.no_grad():
            b, bt = self.assemble(nb)
            pp, pn = self.model([[b]])
            rows = torch.remainder(self.bidx, self.eval_cap) * self.NBmax + self.ar[nb]
            self.pred_store.index_copy_(0, rows, torch.cat([pp, pn], dim=1))
            if self.st is not None:
                self.writeback(bt)
            self.bidx.add_(1)
        return pp, pn

    # ------------------------------------------------------------------ capture
    def capture(self, tr_rng, va_rng, warmup=3):
        snap = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.pool = torch.cuda.graph_pool_handle()
        for phase in ('train', 'eval'):
            for nb in self.buckets:
                if phase == 'train':
                    self.model.train()
                    self.load_phase(tr_rng)
                else:
                    self.model.eval()
                    self.load_phase(va_rng)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(warmup):
                        if phase == 'train':
                            self.opt.zero_grad(set_to_none=True)
                            self.train_body(nb)
                        else:
                            self.eval_body(nb)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                if phase == 'train':
                    self.opt.zero_grad(set_to_none=True)
                    with torch.cuda.graph(g, pool=self.pool):
                        self.train_body(nb)
                    self.g_train[nb] = g
                else:
                    with torch.cuda.graph(g, pool=self.pool):
                        self.eval_body(nb)
                    self.g_eval[nb] = g
                torch.cuda.synchronize()
        self.model.load_state_dict(snap)
        for stt in self.opt.state.values():
            for v in stt.values():
                if torch.is_tensor(v):
                    v.zero_()
        self.reset_state()

    # ------------------------------------------------------------------ run
    def run_train_epoch(self, rng, eager=False):
        self.model.train()
        self.load_phase(rng)
        for i in range(len(rng)):
            nb = int(self.host_bucket[i])
            if eager:
                self.opt.zero_grad(set_to_none=True)
                self.train_body(nb)
            else:
                self.g_train[nb].replay()

    def run_eval(self, rng, eager=False):
        self.model.eval()
        self.load_phase(rng)
        for i in range(len(rng)):
            nb = int(self.host_bucket[i])
            if eager:
                self.eval_body(nb)
            else:
                self.g_eval[nb].replay()
