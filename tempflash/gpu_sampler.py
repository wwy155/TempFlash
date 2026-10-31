# TempFlash: exact GPU re-implementation of TGL's ParallelSampler (sampler_core.cpp)
# for the configuration of every workload of the main table:
#   layer=1, history=1, duration=0, strategy='recent', prop_time=False, k=10.
#
# C++ semantics reproduced (TGL's sampler_core.cpp):
#   update_ts_ptr: for every root (n, t) of the batch, the per-node pointer
#       ptr[n] advances to the last position whose ts <= thr (thr = t - 1e-7f),
#       never moving backwards; with duplicates inside a batch the final pointer
#       is the max over occurrences (the scan is order independent).
#       Closed form: ptr[n] = max(ptr[n], ub(n, thr) - 1), ub = first index in the
#       node's segment with ts > thr (segment end if none).
#   sample_layer (recent): positions k = e, e-1, ..., max(s, e-K)+1 with
#       e = ptr[n] (after the batch update) and s = indptr[n]; keep k iff
#       ts[k] < thr. Output order: roots in input order, newest first.
#   combine_coo: nodes = [roots..., neighbours...], ts = [root ts..., ts[k]...],
#       dts = [0..., t - ts[k]...], row = root index, col = R + neighbour slot.
#
# The upper bound is a single torch.searchsorted over a composite int64 key
# (node << 32 | order-preserving bits of the fp32 timestamp), which is exact for
# fp32 comparisons and needs one kernel instead of a CPU thread pool.
import torch

_EPS = 1e-7


def _ordered_bits(x):
    """Map fp32 values to int64 so that integer order == float order (no NaNs)."""
    b = x.contiguous().view(torch.int32).to(torch.int64)
    o = torch.where(b < 0, b ^ 0x7FFFFFFF, b)        # signed order-preserving
    return o + 2147483648                             # shift into [0, 2^32)


class GPURecentSampler:
    def __init__(self, indptr, indices, eid, ts, num_neighbors, device='cuda',
                 index_dtype=torch.int32):
        dev = torch.device(device)
        self.device = dev
        self.indptr = torch.as_tensor(indptr).to(device=dev, dtype=torch.int64)
        self.indices = torch.as_tensor(indices).to(device=dev, dtype=index_dtype)
        self.eid = torch.as_tensor(eid).to(device=dev, dtype=index_dtype)
        self.ts = torch.as_tensor(ts).to(device=dev, dtype=torch.float32).contiguous()
        self.k = int(num_neighbors)
        self.num_nodes = self.indptr.numel() - 1
        self.seg_start = self.indptr[:-1].contiguous()
        self.seg_end = self.indptr[1:].contiguous()
        deg = self.seg_end - self.seg_start
        seg_node = torch.repeat_interleave(torch.arange(self.num_nodes, device=dev), deg)
        self.keys = (seg_node << 32) | _ordered_bits(self.ts)
        del seg_node
        if self.keys.numel() > 1:
            assert bool((self.keys[1:] >= self.keys[:-1]).all()), \
                'T-CSR timestamps must be non-decreasing inside every node segment'
        self.eps = torch.tensor(_EPS, dtype=torch.float32, device=dev)
        self.ar = torch.arange(self.k, device=dev, dtype=torch.int64)
        # pointer table with one trash slot (index num_nodes) for masked roots
        self.ptr = torch.empty(self.num_nodes + 1, dtype=torch.int64, device=dev)
        self.trash = self.num_nodes
        self.reset()

    def bytes_resident(self):
        n = 0
        for t in (self.indptr, self.indices, self.eid, self.ts, self.keys, self.ptr):
            n += t.numel() * t.element_size()
        return n

    def reset(self):
        """In place (addresses stay valid for captured CUDA graphs)."""
        self.ptr[:self.num_nodes].copy_(self.seg_start)
        self.ptr[self.num_nodes] = 0

    @torch.no_grad()
    def sample_dense(self, nodes, root_ts, root_valid):
        """Static-shape variant: (R, K) window + validity mask, no compaction, no host sync.
        For valid roots the kept slots equal the compacted sample() output, row-major."""
        thr = root_ts - self.eps
        q = (nodes << 32) | _ordered_bits(thr)
        ub = torch.searchsorted(self.keys, q, right=True)
        tgt = torch.where(root_valid, nodes, torch.full_like(nodes, self.trash))
        self.ptr.scatter_reduce_(0, tgt, ub - 1, reduce='amax', include_self=True)
        e = self.ptr.index_select(0, nodes)
        s = self.seg_start.index_select(0, nodes)
        cnt = torch.clamp(e - s, min=0, max=self.k)
        pos = e.unsqueeze(1) - self.ar.unsqueeze(0)
        valid = (self.ar.unsqueeze(0) < cnt.unsqueeze(1)) & root_valid.unsqueeze(1)
        posc = torch.where(valid, pos, torch.zeros_like(pos))
        valid = valid & (self.ts[posc] < thr.unsqueeze(1))
        nb_node = self.indices[posc].to(torch.int64)
        nb_eid = self.eid[posc].to(torch.int64)
        nb_ts = self.ts[posc]
        nb_dt = root_ts.unsqueeze(1) - nb_ts
        return valid, nb_node, nb_eid, nb_ts, nb_dt

    @torch.no_grad()
    def sample(self, nodes, root_ts):
        """nodes: int tensor (R,) on device; root_ts: fp32 (R,) on device."""
        nodes = nodes.to(self.device, torch.int64)
        root_ts = root_ts.to(self.device, torch.float32)
        thr = root_ts - self.eps                                   # fp32 arithmetic
        q = (nodes << 32) | _ordered_bits(thr)
        ub = torch.searchsorted(self.keys, q, right=True)          # first key > q
        # ub lies inside [seg_start, seg_end] by construction of the keys
        self.ptr.scatter_reduce_(0, nodes, ub - 1, reduce='amax', include_self=True)
        e = self.ptr.index_select(0, nodes)
        s = self.seg_start.index_select(0, nodes)
        cnt = torch.clamp(e - s, min=0, max=self.k)
        pos = e.unsqueeze(1) - self.ar.unsqueeze(0)                # (R, K) newest first
        valid = self.ar.unsqueeze(0) < cnt.unsqueeze(1)
        posc = torch.where(valid, pos, torch.zeros_like(pos))
        valid &= self.ts[posc] < thr.unsqueeze(1)
        r_idx, j_idx = valid.nonzero(as_tuple=True)                # row-major order
        sel = posc[r_idx, j_idx]
        R = nodes.numel()
        M = sel.numel()
        nb_nodes = self.indices.index_select(0, sel).to(torch.int64)
        nb_ts = self.ts.index_select(0, sel)
        nb_eid = self.eid.index_select(0, sel).to(torch.int64)
        nb_dts = root_ts.index_select(0, r_idx) - nb_ts
        zeros = torch.zeros(R, dtype=torch.float32, device=self.device)
        return {
            'row': r_idx,
            'slot': j_idx,
            'col': torch.arange(R, R + M, device=self.device, dtype=torch.int64),
            'nodes': torch.cat([nodes, nb_nodes]),
            'ts': torch.cat([root_ts, nb_ts]),
            'dts': torch.cat([zeros, nb_dts]),
            'eid': nb_eid,
            'dim_in': R + M,
            'dim_out': R,
        }
