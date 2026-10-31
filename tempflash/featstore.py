# TempFlash feature/state tiers. Every structure is addressed by GPU index tensors;
# the CPU never gathers rows on the training path.
#   device : fp32 table resident in GPU memory (index_select)
#   packed : lossless bit-packed bool table resident in GPU memory, unpacked on GPU
#   pinned : fp32 table in page-locked host memory, rows fetched by zero-copy GPU
#            gather kernels (exact; used only when the fp32 table exceeds HBM)
import torch
import dgl

_SHIFTS = {}


def _shifts(device):
    key = str(device)
    if key not in _SHIFTS:
        _SHIFTS[key] = torch.arange(8, dtype=torch.uint8, device=device)
    return _SHIFTS[key]


class DeviceTable:
    tier = 'device'

    def __init__(self, t):
        self.t = t.to('cuda').contiguous()
        self.N, self.dim = self.t.shape

    def gather(self, idx):
        return self.t.index_select(0, idx)

    def all(self):
        return self.t

    def nbytes_device(self):
        return self.t.numel() * self.t.element_size()


class PackedTable:
    tier = 'packed'

    def __init__(self, codes, d, dpad):
        self.codes = codes.to('cuda').contiguous()
        self.N, self.dim, self.dpad = codes.shape[0], int(d), int(dpad)
        self.sh = _shifts(self.codes.device)

    def gather(self, idx):
        c = self.codes.index_select(0, idx)
        bits = ((c.unsqueeze(-1) >> self.sh) & 1).to(torch.float32)
        return bits.reshape(-1, self.dpad)[:, :self.dim]

    def nbytes_device(self):
        return self.codes.numel()


class PinnedTable:
    tier = 'pinned'

    def __init__(self, t, dedup=False):
        self.t = t.contiguous()
        self.N, self.dim = self.t.shape
        self._handle = dgl.utils.pin_memory_inplace(self.t)
        self.dedup = dedup

    def gather(self, idx):
        if self.dedup and idx.numel() > 0:
            uni, inv = torch.unique(idx, return_inverse=True)
            return dgl.utils.gather_pinned_tensor_rows(self.t, uni).index_select(0, inv)
        return dgl.utils.gather_pinned_tensor_rows(self.t, idx)

    def nbytes_device(self):
        return 0


def table_nbytes_fp32(N, d):
    return int(N) * int(d) * 4
