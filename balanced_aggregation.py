"""Dataset-balanced aggregation for the pooled federation (FEDALL).

Same folds, seeds, local training and validation-frozen thresholds as study.py. In study.py every
client is weighted by its window count. Here each dataset in the federation (the target's training
subjects and every source dataset) receives the same total weight, and the clients inside a dataset
are weighted by their window counts.

  python balanced_aggregation.py --site kfall --seed 0 --out results_balanced
  python balanced_aggregation.py --summary results_study results_balanced
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import common as C
import study as S

ROUNDS = (15, 60)


def pool_groups(site, cache):
    out = []
    for g, (name, d) in enumerate(S.POOL):
        if name == site:
            continue
        D = cache[(name, d)]
        for s in np.unique(D["subj"]):
            m = D["subj"] == s
            out.append((D["X"][m], D["y"][m], g))
    return out


def balanced_weights(sizes, groups):
    tot = {}
    for n, g in zip(sizes, groups):
        tot[g] = tot.get(g, 0) + n
    return [n / tot[g] / len(tot) for n, g in zip(sizes, groups)]


def run(site, seed, cache, log):
    D = cache[(site, None)]
    subjects = sorted(set(D["subj"].tolist()))
    folds = C.make_folds(subjects, D["ages"], 5, seed=100)
    pool = pool_groups(site, cache)
    rows = []
    for fi, te in enumerate(folds):
        va = folds[(fi + 1) % 5]
        tr = [s for s in subjects if s not in te and s not in va]
        own = [(D["X"][D["subj"] == s], D["y"][D["subj"] == s]) for s in tr if (D["subj"] == s).any()]
        clients = own + [(x, y) for x, y, _ in pool]
        groups = [-1] * len(own) + [g for _, _, g in pool]
        w = balanced_weights([len(c[1]) for c in clients], groups)
        yall = np.concatenate([c[1] for c in clients])
        sd = seed * 1000 + fi
        models = C.fedavg(clients, max(ROUNDS), 1, 3e-3, 64, sd, C.pos_weight(yall), snap=ROUNDS, wts=w)

        def collect(users):
            per = {}
            for u in users:
                mu = D["subj"] == u
                adl = np.where(mu & (D["y"] == 0))[0]
                fall = np.where(mu & (D["y"] == 1))[0]
                if len(adl) < 4:
                    continue
                per[u] = {r: (C.score(models[r], D["X"][adl]),
                              C.score(models[r], D["X"][fall]) if len(fall) else np.zeros(0)) for r in ROUNDS}
            return per

        perv, pert = collect(va), collect(te)
        thr = {r: S.freeze([perv[u][r][1] for u in perv]) for r in ROUNDS}
        for u, rec in pert.items():
            row = dict(site=site, seed=seed, fold=fi, subj=int(u), n_fall=int(len(rec[ROUNDS[0]][1])),
                       n_adl=int(len(rec[ROUNDS[0]][0])), exp_h=float(D["exp"].get(u, np.nan)))
            for r, (a, f) in rec.items():
                row[f"fp_BAL_r{r}"] = int((a >= thr[r]).sum())
                row[f"hit_BAL_r{r}"] = int((f >= thr[r]).sum())
                row[f"auc_BAL_r{r}"] = float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(f))],
                                                           np.r_[a, f])) if len(f) >= 3 and len(a) >= 10 else None
            rows.append(row)
        print(f"[{site}] seed {seed} fold {fi} done", file=log, flush=True)
    return rows


def summarise(study_dir, bal_dir):
    rng = np.random.default_rng(11)
    ref = {}
    for p in glob.glob(os.path.join(study_dir, "study_kfall_s*.json")):
        for r in json.load(open(p))["rows"]:
            ref[(r["seed"], r["fold"], r["subj"])] = r
    mine = {}
    for p in glob.glob(os.path.join(bal_dir, "balanced_*_s*.json")):
        for r in json.load(open(p)):
            mine[(r["seed"], r["fold"], r["subj"])] = r
    keys = sorted(set(ref) & set(mine))
    out = {"n_runs": len(keys), "seeds": sorted({k[0] for k in keys})}
    for rd in ROUNDS:
        users = {}
        for k in keys:
            a, b = ref[k], mine[k]
            v = [a[f"auc_FED_r{rd}"], a[f"auc_FEDALL_r{rd}"], b[f"auc_BAL_r{rd}"]]
            if all(x is not None for x in v):
                users.setdefault(k[2], []).append(v)
        M = np.array([np.mean(v, 0) for v in users.values()])
        def ci(d):
            bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
            return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res = {"n_users": int(len(M)), "auc_FED": float(M[:, 0].mean()), "auc_FEDALL": float(M[:, 1].mean()),
               "auc_BAL": float(M[:, 2].mean()),
               "BAL-FEDALL": [float((M[:, 2] - M[:, 1]).mean()), *ci(M[:, 2] - M[:, 1])],
               "BAL-FED": [float((M[:, 2] - M[:, 0]).mean()), *ci(M[:, 2] - M[:, 0])]}
        fa = {}
        for arm, src, key in (("FEDALL", ref, f"FEDALL_r{rd}"), ("BAL", mine, f"BAL_r{rd}")):
            per = {}
            for k in keys:
                r = src[k]
                if r["n_adl"] >= 10 and r["exp_h"] and r["exp_h"] > 0:
                    per.setdefault(k[2], []).append(r[f"fp_{key}"] / r["exp_h"])
            fa[arm] = float(np.mean([np.mean(v) for v in per.values()]))
        hits = {arm: sum(src[k][f"hit_{key}"] for k in keys) / sum(src[k]["n_fall"] for k in keys)
                for arm, src, key in (("FEDALL", ref, f"FEDALL_r{rd}"), ("BAL", mine, f"BAL_r{rd}"))}
        res["fp_per_h"], res["sens"] = fa, hits
        out[f"r{rd}"] = res
    json.dump(out, open(os.path.join(bal_dir, "balanced_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="kfall")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(C.ROOT, "results_balanced"))
    ap.add_argument("--summary", nargs=2, metavar=("STUDY_DIR", "BALANCED_DIR"))
    a = ap.parse_args()
    if a.summary:
        summarise(*a.summary)
        return
    cache = {k: C.load_site(*k) for k in S.POOL}
    cache.setdefault((a.site, None), C.load_site(a.site))
    rows = run(a.site, a.seed, cache, os.sys.stdout)
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(os.path.join(a.out, f"balanced_{a.site}_s{a.seed}.json"), "w"))
    print("done", a.site, a.seed, len(rows))


if __name__ == "__main__":
    main()
