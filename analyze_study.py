"""Merge study_*.json into the tables used in the paper.

Writes into <dir>:
  rows_{site}_r15.json, rows_{site}_r60.json   per-user rows (arms FED, CEN, LOC, FEDALL) with the
                                               validation-frozen threshold (input of stats_tests.py)
  rows_retro_{site}_r60.json                   the same with the retrospective threshold
  study_summary.json                           AUROC, achieved sensitivity, false-positive windows per hour and percent of windows for
                                               every arm and round, frozen and retrospective
"""
import glob
import json
import os
import sys

import numpy as np

B = 2000
SWEEP_ARMS = ("FED", "CEN", "POOL")


def load(d):
    rows, man, tags = {}, {}, None
    for p in sorted(glob.glob(os.path.join(d, "study_*_s[0-9].json"))):
        z = json.load(open(p))
        tags = z["tags"]
        for r in z["rows"]:
            site = r["site"] + ("" if r["dev"] is None else f"_D{r['dev']}")
            rows.setdefault(site, []).append(r)
        man[os.path.basename(p)] = z["manifest"]
    return rows, man, tags


def stats_rows(rows, rnd, retro):
    out = []
    pre_fp, pre_hit = ("fpr", "hitr") if retro else ("fp", "hit")
    for r in rows:
        o = {k: r[k] for k in ("site", "dev", "seed", "fold", "subj", "n_fall", "n_adl", "exp_h", "age")}
        for arm in ("FED", "CEN", "FEDALL", "LOC"):
            k = arm if arm == "LOC" else f"{arm}_r{rnd}"
            o[f"auc_{arm}"] = r[f"auc_{k}"]
            o[f"fp_{arm}"] = r[f"{pre_fp}_{k}"]
            o[f"nev_{arm}"] = r[f"nev_{k}"]
            nf = r["nfs_LOC"] if arm == "LOC" else r["n_fall"]
            o[f"sens_{arm}"] = r[f"{pre_hit}_{k}"] / nf if nf else None
        out.append(o)
    return out


def auc_population(rows):
    """Users with a defined AUROC for every arm at 60 rounds and for LOC (the paired-comparison population)."""
    need = ("auc_FED_r60", "auc_CEN_r60", "auc_FEDALL_r60", "auc_LOC")
    ok = {}
    for r in rows:
        for k in need:
            if r[k] is not None:
                ok.setdefault(r["subj"], set()).add(k)
    return {u for u, v in ok.items() if len(v) == len(need)}


def boot_ci(v, rng):
    return [float(np.percentile(v[rng.integers(0, len(v), (B, len(v)))].mean(1), q)) for q in (2.5, 97.5)]


def summarise(rows, k, rng, pop):
    subj = {}
    for r in rows:
        subj.setdefault(r["subj"], []).append(r)
    a = np.array([np.mean([x[f"auc_{k}"] for x in v if x[f"auc_{k}"] is not None])
                  for u, v in subj.items() if u in pop])
    ev = [v for v in subj.values() if v[0]["n_adl"] >= 10 and v[0]["exp_h"] and v[0]["exp_h"] > 0]
    fa = np.array([np.mean([x[f"fp_{k}"] / x["exp_h"] for x in v]) for v in ev])
    far = np.array([np.mean([x[f"fpr_{k}"] / x["exp_h"] for x in v]) for v in ev])
    ew = [v for v in subj.values() if v[0]["n_adl"] >= 10]
    fpw = np.array([100 * np.mean([x[f"fp_{k}"] / x[f"nev_{k}"] for x in v if x[f"nev_{k}"] > 0])
                    for v in ew if any(x[f"nev_{k}"] > 0 for x in v)])
    nfk = "nfs_LOC" if k == "LOC" else "n_fall"
    users = [v for v in subj.values() if sum(x[nfk] for x in v) > 0]
    hit = np.array([sum(x[f"hit_{k}"] for x in v) for v in users], float)
    hitr = np.array([sum(x[f"hitr_{k}"] for x in v) for v in users], float)
    nf = np.array([sum(x[nfk] for x in v) for v in users], float)
    idx = rng.integers(0, len(users), (B, len(users)))
    sens_ci = [float(np.percentile(hit[idx].sum(1) / nf[idx].sum(1), q)) for q in (2.5, 97.5)]
    fpr = sum(x[f"fp_{k}"] for r in rows for x in [r]) / sum(x[f"nev_{k}"] for r in rows for x in [r])
    return {"auc": float(a.mean()), "auc_ci": boot_ci(a, rng), "fa_per_h": float(fa.mean()), "fa_ci": boot_ci(fa, rng),
            "fp_pct_windows": float(fpw.mean()), "fp_pct_windows_ci": boot_ci(fpw, rng),
            "sens": float(hit.sum() / nf.sum()), "sens_ci": sens_ci, "fp_per_window": float(fpr),
            "fa_per_h_retro": float(far.mean()), "sens_retro": float(hitr.sum() / nf.sum()),
            "n_users_auc": int(len(a)), "n_users_fa": int(len(fa)), "n_users_sens": int(len(users))}


def main(d):
    rng = np.random.default_rng(7)
    rows, man, tags = load(d)
    summ = {"sites": {}}
    for site, rs in rows.items():
        for rnd in (15, 60):
            if site == "fallalld_D2" or site == "fallalld_D3":
                if rnd == 15:
                    continue
            json.dump(stats_rows(rs, rnd, False), open(os.path.join(d, f"rows_{site}_r{rnd}.json"), "w"))
        json.dump(stats_rows(rs, 60, True), open(os.path.join(d, f"rows_retro_{site}_r60.json"), "w"))
        pop = auc_population(rs)
        summ["sites"][site] = {k: summarise(rs, k, rng, pop) for k in tags}
    json.dump(summ, open(os.path.join(d, "study_summary.json"), "w"), indent=1)
    json.dump(man, open(os.path.join(d, "split_manifest.json"), "w"))
    for site, res in summ["sites"].items():
        print(site)
        for arm in (*SWEEP_ARMS, "FEDALL"):
            cells = [f"r{k.split('_r')[1]}: {v['auc']:.4f} / sens {v['sens']:.2f} / FA {v['fa_per_h']:.1f}"
                     for k, v in res.items() if k.startswith(arm + "_")]
            print(f"  {arm:6s}", " | ".join(cells))
        print(f"  LOC    {res['LOC']['auc']:.4f} / sens {res['LOC']['sens']:.2f} / FA {res['LOC']['fa_per_h']:.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_study"))
