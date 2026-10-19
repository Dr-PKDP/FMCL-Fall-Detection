"""Per-subject exposure table and data-flow table from the cache files (and data_flow.json if present).

python audit.py [out_dir]     reads FMCL_CACHE (default cache/); data_flow.json is read from FMCL_CACHE
"""
import csv
import json
import os
import sys

import numpy as np

import common as C

SETS = [("unimib", None), ("umafall", None), ("fallalld", 1), ("fallalld", 2), ("fallalld", 3), ("kfall", None)]


def main(out):
    os.makedirs(out, exist_ok=True)
    rows = []
    for name, dev in SETS:
        path = os.path.join(C.CACHE_DIR, f"cache_{name}.npz")
        if not os.path.exists(path):
            print(f"{name}: no cache file, skipped")
            continue
        d = np.load(path)
        m = np.ones(len(d["y"]), bool) if dev is None else d["dev"] == dev
        exp = {}
        for k, h in zip(d["exp_key"].tolist(), d["exp_hours"].tolist()):
            if dev is None or k % 10 == dev:
                exp[k // 10] = exp.get(k // 10, 0.0) + h
        for s in sorted(set(d["subj"][m].tolist())):
            ms = m & (d["subj"] == s)
            adl, fall = int((ms & (d["y"] == 0)).sum()), int((ms & (d["y"] == 1)).sum())
            h = exp.get(int(s), 0.0)
            rows.append(dict(dataset=name + ("" if dev is None else f"_D{dev}"), subject=int(s), adl_windows=adl,
                             fall_windows=fall, adl_recording_h=round(h, 4),
                             adl_windows_per_recorded_h=round(adl / h, 1) if h > 0 else ""))
    with open(os.path.join(out, "exposure_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tot = {}
    for r in rows:
        t = tot.setdefault(r["dataset"], dict(subjects=0, adl_windows=0, adl_recording_h=0.0))
        t["subjects"] += 1
        t["adl_windows"] += r["adl_windows"]
        t["adl_recording_h"] += r["adl_recording_h"]
    for k, t in tot.items():
        print(f"{k:12s} subjects {t['subjects']:3d}  ADL windows {t['adl_windows']:6d}  "
              f"recorded ADL {t['adl_recording_h']:6.2f} h  windows per recorded hour {t['adl_windows'] / t['adl_recording_h']:7.1f}")
    ref = os.environ.get("FMCL_REF_CACHE")
    if ref:
        for name in ("unimib", "umafall", "fallalld", "kfall"):
            pa, pb = os.path.join(C.CACHE_DIR, f"cache_{name}.npz"), os.path.join(ref, f"cache_{name}.npz")
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            a, b = np.load(pa), np.load(pb)
            same = all(np.array_equal(a[k], b[k]) for k in ("X", "y", "subj", "trial", "exp_hours"))
            print(f"cache_{name}.npz identical to reference: {same}")
    below = {}
    for name in ("unimib", "umafall", "fallalld", "kfall"):
        path = os.path.join(C.CACHE_DIR, f"cache_{name}.npz")
        if os.path.exists(path):
            d = np.load(path)
            pk = np.linalg.norm(d["X"].astype(np.float32), axis=1).max(1)
            below[name] = int(((d["y"] == 1) & (pk < C.TRIGGER_G)).sum())
    p = os.path.join(C.CACHE_DIR, "data_flow.json")
    if os.path.exists(p):
        flow = json.load(open(p))
        with open(os.path.join(out, "data_flow.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "fall_trials_found", "fall_trials_loaded", "fall_excluded_before_trigger",
                        "fall_windows_below_trigger", "fall_windows_final"])
            for k, v in flow.items():
                w.writerow([k, v.get("fall_trials_found"), v.get("fall_trials_loaded"), v.get("fall_excluded", 0),
                            below.get(k), v.get("fall_windows_built")])
                print(k, {a: v.get(a) for a in ("fall_trials_found", "fall_trials_loaded", "fall_excluded")},
                      "below 1.5 g:", below.get(k), "final windows:", v.get("fall_windows_built") - (below.get(k) or 0))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_study"))
