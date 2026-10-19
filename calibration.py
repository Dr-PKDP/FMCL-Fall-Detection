"""Per-user score calibration against the raw FedAvg score.

Raw score: logit difference of the FedAvg model. Calibrated score: (s - m) / a, where m and a are
the median and scaled MAD (floor 0.2) of the user's own ADL scores in the recording half that does
not contain the query. Every window of a user, falls included, is assigned to a half by recording
and without its label, so one rule scores all windows. A half with fewer than five ADL windows is
replaced by the median and scaled MAD of the training users' ADL scores.

Thresholds target 90% sensitivity on the validation users' falls and are applied to the test users.

  python calibration.py --site unimib --seeds 5
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import common as C

torch.set_num_threads(1)
ARMS = ("raw", "cal")


def scores_for(model, D, users, g_stats, rng):
    per = {}
    for u in users:
        mu = D["subj"] == u
        adl = np.where(mu & (D["y"] == 0))[0]
        fall = np.where(mu & (D["y"] == 1))[0]
        if len(adl) < 4:
            continue
        sa, sf = C.score(model, D["X"][adl]), C.score(model, D["X"][fall]) if len(fall) else np.zeros(0)
        hid = C.split_halves(np.concatenate([adl, fall]).astype(int), D["trial"], rng)
        ha, hf = hid[:len(adl)], hid[len(adl):]
        ca, cf = np.zeros(len(adl)), np.zeros(len(fall))
        for r in (0, 1):
            ref = sa[ha == 1 - r]
            m, a = C.robust_stats(ref, 0.2) if len(ref) >= 5 else g_stats
            ca[ha == r] = (sa[ha == r] - m) / a
            if len(fall):
                cf[hf == r] = (sf[hf == r] - m) / a
        per[u] = dict(raw=(sa, sf), cal=(ca, cf))
    return per


def run_fold(D, tr, te, va, args, seed):
    clients = [(D["X"][D["subj"] == s], D["y"][D["subj"] == s]) for s in tr if (D["subj"] == s).any()]
    yc = np.concatenate([c[1] for c in clients])
    model = C.fedavg(clients, args.rounds, 1, 3e-3, 64, seed, C.pos_weight(yc))
    g_stats = C.robust_stats(C.score(model, np.concatenate([c[0][c[1] == 0] for c in clients])), 0.2)
    perv = scores_for(model, D, va, g_stats, np.random.default_rng(seed + 29))
    pert = scores_for(model, D, te, g_stats, np.random.default_rng(seed + 17))
    thr = {k: float(np.quantile(np.concatenate([perv[u][k][1] for u in perv if len(perv[u][k][1])]),
                                1 - C.TARGET_SENS)) for k in ARMS}
    rows = []
    for u, p in pert.items():
        row = dict(subj=int(u), n_adl=int(len(p["raw"][0])), n_fall=int(len(p["raw"][1])),
                   exp_h=float(D["exp"].get(u, np.nan)))
        for k in ARMS:
            a, f = p[k]
            row[f"fp_{k}"] = int((a >= thr[k]).sum())
            row[f"hit_{k}"] = int((f >= thr[k]).sum())
            row[f"auc_{k}"] = float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(f))], np.r_[a, f])) \
                if len(f) >= 3 and len(a) >= 10 else None
        rows.append(row)
    return rows


def summarise(rows, rng, B=2000):
    by = {}
    for r in rows:
        by.setdefault(r["subj"], []).append(r)
    mean = lambda v: float(np.mean(v))
    fa_users = [v for v in by.values() if v[0]["n_adl"] >= 10 and v[0]["exp_h"] and v[0]["exp_h"] > 0]
    fa = {k: np.array([mean([x[f"fp_{k}"] / x["exp_h"] for x in v]) for v in fa_users]) for k in ARMS}
    idx = rng.integers(0, len(fa_users), (B, len(fa_users)))
    rel = (fa["cal"][idx].mean(1) / fa["raw"][idx].mean(1) - 1) * 100
    pu = [v for v in by.values() if v[0]["n_adl"] >= 10]
    pct = {k: np.array([100 * mean([x[f"fp_{k}"] / x["n_adl"] for x in v]) for v in pu]) for k in ARMS}
    ip = rng.integers(0, len(pu), (B, len(pu)))
    relp = (pct["cal"][ip].mean(1) / pct["raw"][ip].mean(1) - 1) * 100
    auc_users = [v for v in by.values() if all(x[f"auc_{k}"] is not None for x in v for k in ARMS)]
    auc = {k: np.array([mean([x[f"auc_{k}"] for x in v]) for v in auc_users]) for k in ARMS}
    ia = rng.integers(0, len(auc_users), (B, len(auc_users)))
    d = auc["cal"] - auc["raw"]
    res = {"n_users_fa": len(fa_users), "n_users_auc": len(auc_users)}
    for k in ARMS:
        res[f"fa_per_h_{k}"] = float(fa[k].mean())
        res[f"sens_{k}"] = float(sum(r[f"hit_{k}"] for r in rows) / max(1, sum(r["n_fall"] for r in rows)))
        res[f"auc_{k}"] = float(auc[k].mean()) if len(auc_users) else None
    for k in ARMS:
        res[f"fp_pct_windows_{k}"] = float(pct[k].mean())
    res["fp_pct_change"] = [float((pct["cal"].mean() / pct["raw"].mean() - 1) * 100),
                            *[float(np.percentile(relp, q)) for q in (2.5, 97.5)]]
    res["fa_change_pct"] = [float((fa["cal"].mean() / fa["raw"].mean() - 1) * 100),
                            *[float(np.percentile(rel, q)) for q in (2.5, 97.5)]]
    res["auc_change"] = [float(d.mean()), *[float(np.percentile(d[ia].mean(1), q)) for q in (2.5, 97.5)]]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["unimib", "umafall", "fallalld"])
    ap.add_argument("--dev", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(C.ROOT, "results_study"))
    args = ap.parse_args()
    D = C.load_site(args.site, args.dev)
    subjects = sorted(set(D["subj"].tolist()))
    label = args.site + ("" if args.dev is None else f"_D{args.dev}")
    folds = C.make_folds(subjects, D["ages"], 5, seed=100)
    rows, t0 = [], time.time()
    for seed in range(args.seeds):
        for fi, te in enumerate(folds):
            va = folds[(fi + 1) % 5]
            tr = [s for s in subjects if s not in te and s not in va]
            for r in run_fold(D, tr, te, va, args, seed * 1000 + fi):
                rows.append(dict(r, seed=seed, fold=fi))
            print(f"[{label}] seed {seed} fold {fi} [{time.time() - t0:.0f}s]", flush=True)
    res = summarise(rows, np.random.default_rng(0))
    res.update(label=label, rounds=args.rounds, seeds=args.seeds)
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(summary=res, rows=rows), open(os.path.join(args.out, f"calibration_{label}.json"), "w"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
