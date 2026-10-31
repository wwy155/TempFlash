"""Lossless bit-packing of boolean feature tables: 8 boolean dimensions per byte (GDELT's features).
The packed codes are kept on the GPU and unpacked there by featstore.PackedTable."""
import torch


class PackedFeatureTable:
    """codes[i] holds row i of an (N, d) boolean table in ceil(d / 8) bytes; bit j of byte b is dimension
    8 b + j (dimensions beyond d are zero)."""

    def __init__(self, feat_bool, chunk_rows=4_000_000):
        assert feat_bool.dtype == torch.bool, 'PackedFeatureTable requires bool input'
        N, d = feat_bool.shape
        self.N, self.d = N, d
        self.dpad = (d + 7) // 8 * 8
        weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
        self.codes = torch.empty(N, self.dpad // 8, dtype=torch.uint8)
        for i in range(0, N, chunk_rows):
            j = min(i + chunk_rows, N)
            blk = feat_bool[i:j]
            if self.dpad != d:
                blk = torch.cat([blk, torch.zeros(j - i, self.dpad - d, dtype=torch.bool)], dim=1)
            self.codes[i:j] = (blk.reshape(j - i, self.dpad // 8, 8).to(torch.uint8) * weights).sum(dim=2).to(torch.uint8)

    @property
    def shape(self):
        return (self.N, self.d)
