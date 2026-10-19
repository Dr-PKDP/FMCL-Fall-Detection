import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("FMCL_CACHE", os.path.join(ROOT, "cache"))
TRIGGER_G = 1.5
TARGET_SENS = 0.90
N_PARAMS = 13898
torch.set_num_threads(1)


def load_site(name, dev=None):
    d = np.load(os.path.join(CACHE_DIR, f"cache_{name}.npz"))
    m = np.ones(len(d["y"]), bool) if dev is None else (d["dev"] == dev)
    X = d["X"][m].astype(np.float32)
    mag = np.linalg.norm(X, axis=1, keepdims=True)
    X4 = np.clip(np.concatenate([X, mag], axis=1), -8, 8)
    y = d["y"][m].astype(np.int64)
    subj = d["subj"][m].astype(int)
    trial = d["trial"][m].astype(int)
    age = d["age"][m].astype(int)
    peak = mag[:, 0, :].max(axis=1)
    keep = (y == 0) | (peak >= TRIGGER_G)
    exp = {}
    for k, h in zip(d["exp_key"].tolist(), d["exp_hours"].tolist()):
        s, dv = k // 10, k % 10
        if dev is None or dv == dev:
            exp[s] = exp.get(s, 0.0) + h
    ages = {int(s): int(a) for s, a in zip(subj, age)}
    return dict(X=X4[keep], y=y[keep], subj=subj[keep], trial=trial[keep], exp=exp, ages=ages,
                n_untriggered_falls=int(((y == 1) & ~(peak >= TRIGGER_G)).sum()))


def make_folds(subjects, ages, k, seed):
    subjects = sorted(subjects)
    rng = np.random.default_rng(seed)
    if all(ages.get(s, -1) > 0 for s in subjects):
        order = sorted(subjects, key=lambda s: (-ages[s], rng.random()))
    else:
        order = list(rng.permutation(subjects))
    folds = [[] for _ in range(k)]
    for i, s in enumerate(order):
        r = i // k
        f = i % k if r % 2 == 0 else k - 1 - (i % k)
        folds[f].append(int(s))
    return folds


class GateCNN(nn.Module):
    def __init__(self, cin=4, w=24):
        super().__init__()
        self.c1 = nn.Conv1d(cin, w, 7, 2, 3)
        self.g1 = nn.GroupNorm(4, w)
        self.c2 = nn.Conv1d(w, 2 * w, 5, 2, 2)
        self.g2 = nn.GroupNorm(4, 2 * w)
        self.c3 = nn.Conv1d(2 * w, 2 * w, 3, 2, 1)
        self.g3 = nn.GroupNorm(4, 2 * w)
        self.head = nn.Linear(4 * w, 2)

    def embed(self, x):
        x = F.relu(self.g1(self.c1(x)))
        x = F.relu(self.g2(self.c2(x)))
        x = F.relu(self.g3(self.c3(x)))
        return torch.cat([x.mean(-1), x.amax(-1)], dim=1)

    def forward(self, x):
        return self.head(self.embed(x))


def augment(xb):
    return xb * torch.empty(len(xb), 1, 1).uniform_(0.92, 1.08) + 0.01 * torch.randn_like(xb)


def pos_weight(y):
    return float(np.clip((y == 0).sum() / max(1, (y == 1).sum()), 1.0, 10.0))


def fedavg(clients, rounds, epochs, lr, bs, seed, pos_w, snap=None, wts=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = GateCNN()
    work = GateCNN()
    w_loss = torch.tensor([1.0, pos_w])
    saved = {}
    for rd in range(1, rounds + 1):
        acc = None
        tot = 0
        for ci, (X, y) in enumerate(clients):
            work.load_state_dict(g.state_dict())
            opt = torch.optim.Adam(work.parameters(), lr=lr)
            n = len(y)
            for _ in range(epochs):
                perm = rng.permutation(n)
                for i in range(0, n, bs):
                    b = perm[i:i + bs]
                    xb = augment(torch.from_numpy(X[b]))
                    loss = F.cross_entropy(work(xb), torch.from_numpy(y[b]), weight=w_loss)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            w = n if wts is None else wts[ci]
            sd = {k: v.detach().clone() * w for k, v in work.state_dict().items()}
            acc = sd if acc is None else {k: acc[k] + sd[k] for k in acc}
            tot += w
        g.load_state_dict({k: v / tot for k, v in acc.items()})
        if snap is not None and rd in snap:
            saved[rd] = copy.deepcopy(g)
    return saved if snap is not None else g


def train_central(X, y, epochs, lr, bs, seed, pos_w, snap=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = GateCNN()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    w = torch.tensor([1.0, pos_w])
    n = len(y)
    saved = {}
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            xb = augment(torch.from_numpy(X[b]))
            loss = F.cross_entropy(m(xb), torch.from_numpy(y[b]), weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if snap is not None and ep in snap:
            saved[ep] = copy.deepcopy(m)
    return saved if snap is not None else m


@torch.no_grad()
def score(model, X):
    model.eval()
    out = []
    for i in range(0, len(X), 1024):
        lo = model(torch.from_numpy(X[i:i + 1024]))
        out.append((lo[:, 1] - lo[:, 0]).numpy())
    return np.concatenate(out)


def knn_dist(Q, C, k=3, exclude_self=False):
    d = np.sqrt(np.maximum(((Q ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * Q @ C.T), 0))
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    k = min(k, d.shape[1] - (1 if exclude_self else 0))
    return np.sort(d, axis=1)[:, :k].mean(1)


def robust_stats(v, floor):
    med = np.median(v)
    sd = 1.4826 * np.median(np.abs(v - med))
    return med, max(sd, floor)


def split_halves(adl_idx, trial, rng):
    trials = np.unique(trial[adl_idx])
    if len(trials) >= 2:
        perm = rng.permutation(trials)
        half = {int(t): i % 2 for i, t in enumerate(perm)}
        return np.array([half[int(t)] for t in trial[adl_idx]])
    return rng.integers(0, 2, len(adl_idx))
