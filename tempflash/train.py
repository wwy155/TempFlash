# TempFlash training driver for ETC's experimental protocol (link prediction, ten most recent neighbors,
# ETC's adaptive batch split, five epochs, the better validation epoch among the last two, one test
# evaluation). The model code is that of TGL/ETC; the training loop is the static-shape, CUDA-graph-replayed
# TempFlash engine (engine.py).
import argparse
import os
p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--config', required=True)
p.add_argument('--gpu', default='0', help='physical GPU id')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--data_root', default='DATA')
p.add_argument('--cache', default='cache')
p.add_argument('--edge_tier', default='auto', choices=['auto', 'device', 'pinned'])
p.add_argument('--factorize', default='auto', choices=['auto', 'on', 'off'])
p.add_argument('--buckets', type=int, default=4)
p.add_argument('--lr', type=float, default=None)
p.add_argument('--dropout', type=float, default=None)
p.add_argument('--att_dropout', type=float, default=None)
p.add_argument('--epochs', type=int, default=-1)
p.add_argument('--no_capture', action='store_true', help='ablation: same static step, eager launches')
p.add_argument('--no_reorder', action='store_true', help='ablation: TGL operator order (concat + per-row projections)')
p.add_argument('--batch_size', type=int, default=None)
p.add_argument('--max_batches', type=int, default=-1, help='smoke: cap train batches per epoch')
p.add_argument('--skip_eval', action='store_true')
p.add_argument('--val_from', type=int, default=0, help='validate only from this epoch on (selection uses e>2)')
p.add_argument('--tag', default='')
p.add_argument('--out_json', default='')
args = p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import time
import json
import random
import hashlib
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from modules import GeneralModel
from utils import parse_config
from gpu_sampler import GPURecentSampler
import fused
import shared
import corelib
import engine

shared.install()
DEV = torch.device('cuda:0')
GB = 2 ** 30


def log(*a):
    print(*a, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def code_hash():
    h = hashlib.sha256()
    for f in ('train.py', 'engine.py', 'gpu_sampler.py', 'fused.py', 'featstore.py', 'corelib.py', 'shared.py'):
        h.update(open(f, 'rb').read())
    return h.hexdigest()[:16]


if __name__ == '__main__':
    set_seed(args.seed)
    tag = args.tag or '{}_{}_s{}'.format(args.data, os.path.basename(args.config).split('.')[0], args.seed)
    log('[tempflash] args={} code={}'.format(vars(args), code_hash()))
    sp, mp, gp, tp = parse_config(args.config)
    for k in ('lr', 'dropout', 'att_dropout', 'batch_size'):
        if getattr(args, k) is not None:
            tp[k] = getattr(args, k)
    shared.ENABLED = not args.no_reorder
    if args.epochs > 0:
        tp['epoch'] = args.epochs
    log('[tempflash] train_param={} memory={} gnn={}'.format(tp, mp.get('memory_update', 'none'), gp['arch']))
    assert sp['layer'] == 1 and sp['history'] == 1 and sp['strategy'] == 'recent'
    assert not sp['prop_time'] and float(sp['duration']) == 0.0
    K = int(sp['neighbor'][0])

    t_load = time.time()
    SRC, DST, TIME, EID, ROLL = corelib.load_events(args.data, args.data_root, args.cache)
    E = len(SRC)
    train_end = int(np.flatnonzero(ROLL > 0)[0])
    val_end = int(np.flatnonzero(ROLL > 1)[0])
    node_t, edge_t, dim_node, dim_edge = corelib.load_features(args.data, args.data_root, args.edge_tier, args.cache,
                                                               num_edges=E)
    g = np.load('{}/{}/ext_full.npz'.format(args.data_root, args.data))
    sampler = GPURecentSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32), K)
    N = sampler.num_nodes
    del g
    tr_rng, va_rng, te_rng = corelib.etc_ranges(args.data, args.data_root, SRC, DST, train_end, val_end,
                                                tp['batch_size'], args.cache)
    # compact resident event arrays: int32 ids when they fit; the event id column is
    # dropped when it is the row index (true for every ETC-preprocessed dataset)
    eid_identity = bool(np.array_equal(EID, np.arange(E, dtype=EID.dtype)))
    idt = torch.int32 if max(int(SRC.max()), int(DST.max()), E) < 2 ** 31 - 1 else torch.int64
    events = (torch.from_numpy(SRC.astype(np.int64)).to(DEV).to(idt),
              torch.from_numpy(DST.astype(np.int64)).to(DEV).to(idt),
              torch.from_numpy(TIME).to(DEV),
              None if eid_identity else torch.from_numpy(EID).to(DEV).to(idt))
    del SRC, DST, TIME, EID
    sizes_all = np.concatenate([r[:, 1] - r[:, 0] for r in (tr_rng, va_rng, te_rng)])
    buckets = engine.make_buckets(sizes_all, args.buckets)
    if args.max_batches > 0:
        tr_rng = tr_rng[:args.max_batches]

    model = GeneralModel(dim_node, dim_edge, sp, mp, gp, tp).to(DEV)
    st = None
    if mp['type'] != 'none':
        full = engine.StateTables.full_bytes(mp, N, dim_edge)
        fact = args.factorize == 'on' or (args.factorize == 'auto' and full > 8 * GB)
        st = engine.StateTables(mp, N, dim_edge, DEV, factorize=fact)
    optimizer = torch.optim.Adam(model.parameters(), lr=tp['lr'], capturable=True)
    use_node = not (gp['arch'] == 'identity' and not mp.get('combine_node_feature', False))
    n_slots = max(len(tr_rng), len(va_rng), len(te_rng))
    eval_cap = max(len(va_rng), len(te_rng))
    eng = engine.Engine(model, optimizer, sampler, node_t, edge_t, events, (sp, mp, gp, tp), st, buckets,
                          n_slots, eval_cap, DEV, use_node_feats=use_node, reorder=not args.no_reorder)
    t_setup = time.time() - t_load
    log('[tempflash] setup {:.1f}s | N={} E={} batches tr/va/te={}/{}/{} buckets={} | node={} edge={} | '
        'T-CSR {:.3f} GB state {:.3f} GB (factorized={}) | allocated {:.3f} GB'.format(
            t_setup, N, E, len(tr_rng), len(va_rng), len(te_rng), buckets,
            None if node_t is None else node_t.tier, None if edge_t is None else edge_t.tier,
            sampler.bytes_resident() / GB, 0 if st is None else st.nbytes() / GB,
            None if st is None else st.factorize, torch.cuda.memory_allocated() / GB))

    t_cap = 0.0
    if not args.no_capture:
        t0 = time.time()
        eng.capture(tr_rng, va_rng)
        torch.cuda.synchronize()
        t_cap = time.time() - t0
        log('[tempflash] captured {} graphs in {:.1f}s | allocated {:.3f} GB reserved {:.3f} GB'.format(
            2 * len(buckets), t_cap, torch.cuda.memory_allocated() / GB, torch.cuda.memory_reserved() / GB))

    def evaluate(rng):
        eng.run_eval(rng, eager=args.no_capture)
        sz = (np.asarray(rng)[:, 1] - np.asarray(rng)[:, 0])
        aps = engine.batched_ap(eng.pred_store, sz, eng.NBmax)
        return float(aps.mean()), sz

    os.makedirs('models', exist_ok=True)
    path_saver = 'models/{}.pkl'.format(tag)
    best_ap, best_e, hist = 0.0, -1, []
    for e in range(tp['epoch']):
        eng.reset_state()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        eng.run_train_epoch(tr_rng, eager=args.no_capture)
        torch.cuda.synchronize()
        t_ep = time.time() - t0
        rec = {'epoch': e, 'time': t_ep, 'loss': float(eng.tot_loss),
               'peak_alloc_gb': torch.cuda.max_memory_allocated() / GB,
               'peak_reserved_gb': torch.cuda.max_memory_reserved() / GB}
        msg = 'Epoch {:d}: total time:{:.2f}s batches:{} loss:{:.1f} peak_alloc:{:.2f}GB'.format(
            e, t_ep, len(tr_rng), rec['loss'], rec['peak_alloc_gb'])
        if not args.skip_eval and e >= min(args.val_from, tp['epoch'] - 1):
            t1 = time.time()
            ap, _ = evaluate(va_rng)
            rec.update(val_ap=ap, eval_time=time.time() - t1)
            if (e > 2 or tp['epoch'] == 1) and ap > best_ap:
                best_e, best_ap = e, ap
                torch.save(model.state_dict(), path_saver)
            msg += ' | val ap:{:.6f} (eval {:.1f}s)'.format(ap, rec['eval_time'])
        log(msg)
        hist.append(rec)
    summ = {'tag': tag, 'data': args.data, 'config': args.config, 'seed': args.seed, 'train_param': tp,
            'buckets': buckets, 'setup_s': t_setup, 'capture_s': t_cap, 'code': code_hash(),
            'epoch_time_mean': float(np.mean([h['time'] for h in hist])), 'epochs': hist,
            'no_capture': args.no_capture, 'edge_tier': None if edge_t is None else edge_t.tier,
            'factorized': None if st is None else st.factorize, 'reorder': not args.no_reorder,
            'shared_paths': sorted(shared.STATS.keys())}
    if not args.skip_eval:
        log('Loading model at epoch {}...'.format(best_e))
        model.load_state_dict(torch.load(path_saver))
        eng.reset_state()
        if st is not None:
            eng.run_eval(tr_rng, eager=args.no_capture)
            eng.run_eval(va_rng, eager=args.no_capture)
        ap, sz = evaluate(te_rng)
        store = eng.pred_store.cpu()
        sk = []
        for i, n in enumerate(sz):
            blk = store[i * eng.NBmax:i * eng.NBmax + int(n)]
            y_pred = torch.cat([blk[:, 0], blk[:, 1]]).sigmoid()
            y_true = torch.cat([torch.ones(int(n)), torch.zeros(int(n))])
            sk.append(average_precision_score(y_true, y_pred))
        ap_sk = float(np.mean(sk))
        log('\ttest AP:{:.6f}  (gpu-AP {:.6f}, |diff| {:.2e})'.format(ap_sk, ap, abs(ap_sk - ap)))
        summ.update(best_epoch=best_e, best_val_ap=best_ap, test_ap=ap_sk, test_ap_gpu=ap)
    log('[tempflash] SUMMARY ' + json.dumps(summ))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(summ, f, indent=1)
