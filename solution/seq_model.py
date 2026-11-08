"""V32 sequence model: train and predict, leave one game out (rules in solution/run_v32.py).

Needs only numpy and torch, so it runs in the GPU environment:
    data/tools/media_env/Scripts/python.exe solution/seq_model.py
Reads <out>/features/*.npz (from `run_v32 export` or `run_v34 export`) and writes one prediction
file per held-out game and test copy to <out>/predictions/. Defaults are V32's.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parents[1]/'data/runs/v32_sequence'
WINDOW, BATCH, STEPS, LR, WD, SEED = 1000, 16, 4000, 1e-3, 1e-4, 3201
SHORT_S, SHORT_WEIGHT = 5.0, 3.0


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, 128), nn.GELU())
        self.gru = nn.GRU(128, 96, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(192, 1)

    def forward(self, x):
        return self.out(self.gru(self.inp(x))[0]).squeeze(-1)


def period_runs(period):
    edges = np.r_[0, np.flatnonzero(np.diff(period)) + 1, len(period)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if period[a] > 0]


def targets(labels, period, fps):
    """y (1 home, 0 away, -1 ignore) and loss weights (x3 inside true possessions under 5 s)."""
    y = np.where(labels == 'H', 1, np.where(labels == 'A', 0, -1)).astype(np.int8)
    w = np.ones(len(labels), np.float32)
    for a, b in period_runs(period):
        local = labels[a:b]
        edges = np.r_[0, np.flatnonzero(local[1:] != local[:-1]) + 1, len(local)]
        for s, e in zip(edges[:-1], edges[1:]):
            if local[s] in ('H', 'A') and (e - s) / fps < SHORT_S:
                w[a + s:a + e] = SHORT_WEIGHT
    return y, w


def load(path):
    d = np.load(path)
    return {'xh': d['x_home'], 'xa': d['x_away'], 'period': d['period'], 'fps': int(d['fps']),
            'labels': d['controlled'], **({'dead': d['dead']} if 'dead' in d.files else {})}


def scaler(copies):
    """Mean/std per column from training copies only (both orientations share columns)."""
    rows = np.concatenate([c[k][c['period'] > 0][::10].astype(np.float32) for c in copies for k in ('xh', 'xa')])
    mean, std = np.nanmean(rows, axis=0), np.nanstd(rows, axis=0)
    return np.nan_to_num(mean), np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)


def prepare(x1, x2, mean, std, device):
    """[x1 | x2] scaled, NaN -> 0 plus a missing flag per column."""
    x = torch.from_numpy(np.concatenate([x1, x2], axis=-1).astype(np.float32)).to(device)
    m = torch.from_numpy(np.r_[mean, mean]).to(device)
    s = torch.from_numpy(np.r_[std, std]).to(device)
    missing = torch.isnan(x)
    x = torch.clamp(torch.nan_to_num((x - m) / s, nan=0.0), -5, 5)
    return torch.cat([x, missing.float()], dim=-1)


def train_fold(train_copies, fold, device):
    torch.manual_seed(SEED + fold)
    rng = np.random.RandomState(SEED + fold)
    mean, std = scaler(train_copies)
    for c in train_copies:
        c['y'], c['w'] = targets(c['labels'], c['period'], c['fps'])
    spans = [(i, a, b) for i, c in enumerate(train_copies) for a, b in period_runs(c['period']) if b - a > WINDOW]
    sizes = np.array([b - a - WINDOW for _, a, b in spans], float)
    net = Net(4 * train_copies[0]['xh'].shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    for step in range(STEPS):
        xs1, xs2, ys, ws = [], [], [], []
        for k in rng.choice(len(spans), BATCH, p=sizes / sizes.sum()):
            i, a, b = spans[k]
            s = a + rng.randint(0, b - a - WINDOW)
            c = train_copies[i]
            y, w = c['y'][s:s + WINDOW].astype(np.float32), c['w'][s:s + WINDOW]
            if rng.rand() < 0.5:  # home/away swap
                xs1.append(c['xa'][s:s + WINDOW]); xs2.append(c['xh'][s:s + WINDOW]); y = np.where(y >= 0, 1 - y, -1)
            else:
                xs1.append(c['xh'][s:s + WINDOW]); xs2.append(c['xa'][s:s + WINDOW])
            ys.append(y); ws.append(w)
        x = prepare(np.stack(xs1), np.stack(xs2), mean, std, device)
        y = torch.from_numpy(np.stack(ys)).to(device)
        w = torch.from_numpy(np.stack(ws)).to(device) * (y >= 0)
        loss = (nn.functional.binary_cross_entropy_with_logits(net(x), y.clamp(min=0), reduction='none') * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        if step % 500 == 0:
            print(f'  fold {fold} step {step} loss {loss.item():.4f}', flush=True)
    return net, mean, std


@torch.no_grad()
def predict(net, mean, std, c, device):
    net.eval()
    """Same 40 s windows as training, 50% overlap; each frame is taken from a window's middle half."""
    p = np.full(len(c['period']), 0.5)
    half, quarter = WINDOW // 2, WINDOW // 4
    for a, b in period_runs(c['period']):
        starts = sorted({*range(a, max(a, b - WINDOW) + 1, half), max(a, b - WINDOW)})
        for s in starts:
            e = min(s + WINDOW, b)
            one = torch.sigmoid(net(prepare(c['xh'][None, s:e], c['xa'][None, s:e], mean, std, device)))[0]
            two = torch.sigmoid(net(prepare(c['xa'][None, s:e], c['xh'][None, s:e], mean, std, device)))[0]
            local = ((one + 1 - two) / 2).cpu().numpy()
            lo = 0 if s == a else quarter
            hi = e - s if e == b else e - s - quarter
            p[s + lo:s + hi] = local[lo:hi]
    return np.clip(p, 1e-3, 1 - 1e-3)


def final(args, device):
    """The submission's model: every practice game, seed SEED + 10, weights saved for numpy inference."""
    feats = args.out/'features'
    games = sorted(p.name[:-len(f'_{args.train[0]}.npz')] for p in feats.glob(f'*_{args.train[0]}.npz'))
    copies = [load(feats/f'{g}_{v}.npz') for g in games for v in args.train]
    columns = np.load(feats/f'{games[0]}_{args.train[0]}.npz')['columns']
    net, mean, std = train_fold(copies, 10, device)
    weights = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    np.savez(args.final, mean=mean, std=std, columns=columns, games=np.array(games), window=WINDOW,
             **{f'w__{k}': v for k, v in weights.items()})
    print(json.dumps({'final': str(args.final), 'games': games}), flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--train', nargs='+', default=['recorded', 'train_edge'])
    parser.add_argument('--test', nargs='+', default=['test_edge'])
    parser.add_argument('--final', type=Path, help='Train once on all games and save weights here (numpy .npz)')
    parser.add_argument('only', nargs='*', help='Limit to these target games')
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    device = 'cuda'
    if args.final:
        final(args, device)
        return
    feats = args.out/'features'
    first = args.test[0]
    games = sorted(p.name[:-len(f'_{first}.npz')] for p in feats.glob(f'*_{first}.npz'))
    if len(games) != 10:
        raise ValueError(f'Expected 10 games, found {len(games)}')
    (args.out/'predictions').mkdir(parents=True, exist_ok=True)
    name = (lambda g, v: f'{g}.npz') if args.test == ['test_edge'] else (lambda g, v: f'{g}_{v}.npz')  # V32 names
    for fold, target in enumerate(games):
        paths = {v: args.out/'predictions'/name(target, v) for v in args.test}
        if all(p.exists() for p in paths.values()) or (args.only and target not in args.only):
            continue
        began = time.time()
        train_games = [g for g in games if g != target]
        copies = [load(feats/f'{g}_{v}.npz') for g in train_games for v in args.train]
        net, mean, std = train_fold(copies, fold, device)
        del copies
        for v, path in paths.items():
            p = predict(net, mean, std, load(feats/f'{target}_{v}.npz'), device)
            np.savez(path, p_home=p, train_games=np.array(train_games))
        print(json.dumps({'fold': fold, 'target': target, 'seconds': round(time.time() - began)}), flush=True)
        del net
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
