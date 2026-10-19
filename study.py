"""Leave-subjects-out study for one dataset and one seed.

Test fold i is scored with models trained on the three folds that are neither the test fold nor
the validation fold (i + 1) mod 5. Each arm's alarm threshold is fixed on the validation
subjects (90% of their fall windows) and applied to the test subjects; the threshold from the
test subjects' own falls is stored as well.

Arms (checkpoints in rounds or epochs)
  FED     FedAvg over the target dataset's training subjects      5 15 30 60 120
  CEN     the same model trained on the pooled windows            5 15 30 60 120
  POOL    FedAvg over the other datasets only                     5 15 30 60 120
  FEDALL  FedAvg over the training subjects and the other datasets  15 60
  LOC     3-NN novelty detector on the user's own activity windows
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

import common as C

POOL = [("unimib", None), ("umafall", None), ("fallalld", 1)]


def other_clients(site, cache):
    out = []
    for name, d in POOL:
        if name == site:
            continue
        D = cache[(name, d)]
        for s in np.unique(D["subj"]):
            m = D["subj"] == s
            out.append((D["X"][m], D["y"][m]))
    return out


def local_scores(Vall, D, adl, fall, rng):
    """Novelty score of every query window against the user's own ADL windows.

    All of a user's recordings, falls included, are assigned to one of two halves by recording and
    without looking at the label. A query in half h is scored against the reference bank made of the
    ADL windows of the other half, so one rule applies to every window. Queries whose bank has fewer
    than four windows are not scored.
    """
    q = np.concatenate([adl, fall]).astype(int)
    hid = C.split_halves(q, D["trial"], rng)
    ha, hf = hid[:len(adl)], hid[len(adl):]
    s_adl = np.full(len(adl), np.nan)
    s_fall = np.full(len(fall), np.nan)
    for r in (0, 1):
        cal = adl[ha == r]
        if len(cal) < 4:
            continue
        Vc = Vall[cal]
        loo = C.knn_dist(Vc, Vc, 3, exclude_self=True)
        mb, sb = C.robust_stats(loo, 0.05 * float(np.median(loo)) + 1e-6)
        ea, ef = np.where(ha != r)[0], np.where(hf != r)[0]
        if len(ea):
            s_adl[ea] = (C.knn_dist(Vall[adl[ea]], Vc, 3) - mb) / sb
        if len(ef):
            s_fall[ef] = (C.knn_dist(Vall[fall[ef]], Vc, 3) - mb) / sb
    return s_adl[~np.isnan(s_adl)], s_fall[~np.isnan(s_fall)]


SWEEP = (5, 15, 30, 60, 120)
ALL_ARM_ROUNDS = (15, 60)


def tags():
    t = [f"{a}_r{r}" for a in ("FED", "CEN", "POOL") for r in SWEEP]
    t += [f"FEDALL_r{r}" for r in ALL_ARM_ROUNDS]
    return t + ["LOC"]


def freeze(vals):
    vals = [v for v in vals if len(v)]
    return float(np.quantile(np.concatenate(vals), 1 - C.TARGET_SENS)) if vals else float("inf")


def run_site(site, dev, seed, cache, log, scores_out):
    D = cache[(site, dev)]
    subjects = sorted(set(D["subj"].tolist()))
    folds = C.make_folds(subjects, D["ages"], 5, seed=100)
    pool_clients = other_clients(site, cache)
    ypool = np.concatenate([c[1] for c in pool_clients])
    Vall = D["X"].reshape(len(D["X"]), -1)
    top = max(SWEEP)
    t0 = time.time()
    pool_models = C.fedavg(pool_clients, top, 1, 3e-3, 64, seed * 1000 + 999, C.pos_weight(ypool), snap=SWEEP)
    print(f"[{site}] seed {seed} pool done [{time.time() - t0:.0f}s]", file=log, flush=True)
    rows, manifest = [], []
    for fi, te in enumerate(folds):
        va = folds[(fi + 1) % 5]
        tr = [s for s in subjects if s not in te and s not in va]
        manifest.append(dict(fold=fi, test=[int(s) for s in te], val=[int(s) for s in va],
                             train=[int(s) for s in tr]))
        clients = [(D["X"][D["subj"] == s], D["y"][D["subj"] == s]) for s in tr if (D["subj"] == s).any()]
        Xc = np.concatenate([c[0] for c in clients])
        yc = np.concatenate([c[1] for c in clients])
        sd = seed * 1000 + fi
        all_clients = clients + pool_clients
        yall = np.concatenate([c[1] for c in all_clients])
        fed = C.fedavg(clients, top, 1, 3e-3, 64, sd, C.pos_weight(yc), snap=SWEEP)
        cen = C.train_central(Xc, yc, top, 3e-3, 64, sd, C.pos_weight(yc), snap=SWEEP)
        fall_ = C.fedavg(all_clients, max(ALL_ARM_ROUNDS), 1, 3e-3, 64, sd, C.pos_weight(yall), snap=ALL_ARM_ROUNDS)
        models = {}
        for r in SWEEP:
            models[f"FED_r{r}"], models[f"CEN_r{r}"], models[f"POOL_r{r}"] = fed[r], cen[r], pool_models[r]
        for r in ALL_ARM_ROUNDS:
            models[f"FEDALL_r{r}"] = fall_[r]

        def collect(users, rng):
            per = {}
            for u in users:
                mu = D["subj"] == u
                adl = np.where(mu & (D["y"] == 0))[0]
                fall = np.where(mu & (D["y"] == 1))[0]
                if len(adl) < 4:
                    continue
                rec = {k: (C.score(m, D["X"][adl]), C.score(m, D["X"][fall]) if len(fall) else np.zeros(0))
                       for k, m in models.items()}
                rec["LOC"] = local_scores(Vall, D, adl, fall, rng)
                rec["n_fall"] = len(fall)
                per[u] = rec
            return per

        perv = collect(va, np.random.default_rng(sd + 29))
        pert = collect(te, np.random.default_rng(sd + 17))
        thr = {k: freeze([perv[u][k][1] for u in perv]) for k in tags()}
        retro = {k: freeze([pert[u][k][1] for u in pert]) for k in tags()}
        for u, rec in pert.items():
            row = dict(site=site, dev=dev, seed=seed, fold=fi, subj=int(u), n_fall=int(rec["n_fall"]), nfs_LOC=int(len(rec["LOC"][1])),
                       n_adl=int(len(rec["FED_r15"][0])), exp_h=float(D["exp"].get(u, np.nan)),
                       age=int(D["ages"].get(u, -1)))
            for k in tags():
                a, f = rec[k]
                row[f"fp_{k}"] = int((a >= thr[k]).sum())
                row[f"hit_{k}"] = int((f >= thr[k]).sum())
                row[f"fpr_{k}"] = int((a >= retro[k]).sum())
                row[f"hitr_{k}"] = int((f >= retro[k]).sum())
                row[f"nev_{k}"] = int(len(a))
                ok = len(f) >= 3 and len(a) >= 10
                row[f"auc_{k}"] = float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(f))],
                                                      np.r_[a, f])) if ok else None
                if scores_out is not None and k in ("FED_r60", "CEN_r60", "FEDALL_r60", "LOC"):
                    scores_out[f"s{seed}_f{fi}_u{u}_{k}_adl"] = a.astype(np.float16)
                    scores_out[f"s{seed}_f{fi}_u{u}_{k}_fall"] = f.astype(np.float16)
            rows.append(row)
        print(f"[{site}] seed {seed} fold {fi} [{time.time() - t0:.0f}s]", file=log, flush=True)
    return rows, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["unimib", "umafall", "fallalld", "kfall"])
    ap.add_argument("--dev", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(C.ROOT, "results_study"))
    ap.add_argument("--save-scores", action="store_true")
    a = ap.parse_args()
    cache = {k: C.load_site(*k) for k in POOL}
    if (a.site, a.dev) not in cache:
        cache[(a.site, a.dev)] = C.load_site(a.site, a.dev)
    tag = a.site + ("" if a.dev is None else f"_D{a.dev}")
    sc = {} if a.save_scores else None
    rows, manifest = run_site(a.site, a.dev, a.seed, cache, sys.stdout, sc)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"study_{tag}_s{a.seed}.json"), "w") as f:
        json.dump({"tags": tags(), "rows": rows, "manifest": manifest}, f)
    if sc is not None:
        np.savez_compressed(os.path.join(a.out, f"scores_{tag}_s{a.seed}.npz"), **sc)
    print("done", tag, len(rows), flush=True)


if __name__ == "__main__":
    main()
