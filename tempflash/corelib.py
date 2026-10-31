# Shared TempFlash helpers (no argparse): ETC adaptive split (verbatim), cached
# event loading, and feature tiers that honour ETC's feature protocol.
import os
import time
import numpy as np
import pandas as pd
import torch
from utils import load_feat
from featstore import DeviceTable, PackedTable, PinnedTable, table_nbytes_fp32


def log(*a):
    print(*a, flush=True)


def get_staleness_constraint(data, batch_size):          # verbatim from ETC train.py
    range_ = np.arange(0, len(data), batch_size)
    if range_[-1] != len(data):
        range_ = np.append(range_, [len(data)])
    staleness = []
    for i in range(len(range_)):
        if i != len(range_) - 1:
            update = 2 * (range_[i + 1] - range_[i])
            nodes = len(np.unique(data[range_[i]:range_[i + 1]]))
            staleness.append(update - nodes)
    return max(np.array(staleness))


def adaptive_split(threshold, data, group_index):        # verbatim from ETC train.py
    start_list = []
    start = 0
    while start < len(data) - 1:
        count_nodes = 0
        count_edges = 0
        node_set = set()
        for i in range(start, len(data)):
            count_edges += 2
            for j in data[i]:
                if j not in node_set:
                    node_set.add(j)
                    count_nodes += 1
            if count_edges - count_nodes > threshold or i == len(data) - 1:
                end = i
                break
        start_list.append(end)
        start = end
    start_list = [0] + start_list
    start_list[-1] = len(data)
    split = []
    for i in range(len(start_list)):
        if i != len(start_list) - 1:
            split.append(range(start_list[i], start_list[i + 1]))
    for i in range(len(split)):
        group_index[split[i]] = i
    return group_index


def group_ranges(group_index, offset):
    g = np.asarray(group_index)
    cut = np.flatnonzero(np.diff(g)) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(g)]])
    assert np.all(np.diff(g[starts]) > 0), 'non-contiguous ETC groups'
    return np.stack([starts + offset, ends + offset], axis=1)


def etc_ranges(data, data_root, src, dst, train_end, val_end, batch_size, cache):
    key = os.path.join(cache, 'splits_{}_{}_{}.npz'.format(os.path.basename(data_root), data, batch_size))
    if os.path.exists(key):
        z = np.load(key)
        return z['train'], z['val'], z['test']
    t0 = time.time()
    out = []
    for lo, hi in ((0, train_end), (train_end, val_end), (val_end, len(src))):
        d = np.stack([src[lo:hi], dst[lo:hi]]).T
        thr = get_staleness_constraint(d, batch_size)
        gi = adaptive_split(thr, d, np.arange(lo, hi))
        out.append(group_ranges(gi, lo))
    os.makedirs(cache, exist_ok=True)
    np.savez(key, train=out[0], val=out[1], test=out[2])
    log('[core] ETC adaptive split computed in {:.1f}s and cached -> {}'.format(time.time() - t0, key))
    return out


def load_events(data, data_root, cache):
    key = os.path.join(cache, 'events_{}_{}.npz'.format(os.path.basename(data_root), data))
    if os.path.exists(key):
        z = np.load(key)
        return z['src'], z['dst'], z['time'], z['eid'], z['roll']
    df = pd.read_csv('{}/{}/edges.csv'.format(data_root, data), usecols=['Unnamed: 0', 'src', 'dst', 'time', 'ext_roll'])
    out = (df['src'].values.astype(np.int64), df['dst'].values.astype(np.int64),
           df['time'].values.astype(np.float32), df['Unnamed: 0'].values.astype(np.int64),
           df['ext_roll'].values.astype(np.int8))
    os.makedirs(cache, exist_ok=True)
    np.savez(key, src=out[0], dst=out[1], time=out[2], eid=out[3], roll=out[4])
    return out


def load_features(data, data_root, edge_tier, cache, num_edges=None):
    """(node_table, edge_table, dim_node, dim_edge). GDELT: bool features, bit-packed
    losslessly; LastFM/Wiki-Talk/Stack-Overflow: ETC's random-feature protocol."""
    if data == 'GDELT':
        nf = '{}/{}/node_features.pt'.format(data_root, data)
        ef = '{}/{}/edge_features.pt'.format(data_root, data)
        raw_node = torch.load(nf)
        node = DeviceTable(raw_node.to(torch.float32))
        ck = os.path.join(cache, 'gdelt_edge_codes.pt')
        if os.path.exists(ck):
            z = torch.load(ck)
            codes, dim, dpad = z['codes'], z['d'], z['dpad']
        else:
            from bitpack import PackedFeatureTable
            raw_edge = torch.load(ef)
            assert raw_edge.dtype == torch.bool
            p = PackedFeatureTable(raw_edge)
            del raw_edge
            codes, dim, dpad = p.codes, p.d, p.dpad
            torch.save({'codes': codes, 'd': dim, 'dpad': dpad}, ck)
        edge = PackedTable(codes, dim, dpad)
        return node, edge, node.dim, edge.dim
    node_feats, edge_feats = load_feat(data, root=data_root)
    print('[core] features from {}/{}: node={} edge={}'.format(data_root, data, None if node_feats is None else tuple(node_feats.shape), None if edge_feats is None else tuple(edge_feats.shape)), flush=True)
    if data == 'WIKITALK' and edge_feats is None:
        # ETC Table 1 lists d_e = 172 for Wiki-Talk and ETC's memory branch indexes
        # edge_feats unconditionally (train.py L347), but the released loader only
        # draws node features. We draw the edge features with the same random
        # protocol ETC uses for Stack-Overflow (as for the other systems).
        assert num_edges is not None
        edge_feats = torch.randn(int(num_edges), 172)
    node = DeviceTable(node_feats) if node_feats is not None else None
    edge = None
    if edge_feats is not None:
        tier = edge_tier
        if tier == 'auto':
            free = torch.cuda.mem_get_info()[0]
            tier = 'device' if table_nbytes_fp32(*edge_feats.shape) < 0.55 * free else 'pinned'
        edge = DeviceTable(edge_feats) if tier == 'device' else PinnedTable(edge_feats)
    return node, edge, (0 if node is None else node.dim), (0 if edge is None else edge.dim)
