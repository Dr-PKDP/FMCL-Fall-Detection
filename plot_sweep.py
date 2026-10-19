"""Figure: AUROC and false-positive windows per recorded hour (UniMiB: percent of activity windows) against communication per participant."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SITES = [("unimib", "UniMiB"), ("umafall", "UMAFall"), ("fallalld_D1", "FallAllD D1"), ("kfall", "KFall")]
STYLE = {"FED": ("tab:blue", "o"), "CEN": ("tab:orange", "s"), "POOL": ("tab:green", "^")}
ROUNDS = [5, 15, 30, 60, 120]
MODEL_BYTES = 55592


def main(res_dir, out):
    d = json.load(open(os.path.join(res_dir, "study_summary.json")))["sites"]
    mb = [2 * r * MODEL_BYTES / 1e6 for r in ROUNDS]
    fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex=True)
    for j, (k, name) in enumerate(SITES):
        for arm, (c, m) in STYLE.items():
            y = [d[k][f"{arm}_r{r}"] for r in ROUNDS]
            ax[0, j].plot(mb, [v["auc"] for v in y], color=c, marker=m, ms=3, lw=1, label=arm)
            ax[1, j].plot(mb, [v["fp_pct_windows" if k == "unimib" else "fa_per_h"] for v in y], color=c, marker=m, ms=3, lw=1)
        ax[0, j].set_title(name, fontsize=8)
        ax[1, j].set_xscale("log")
        ax[1, j].set_xlabel("MB per participant", fontsize=7)
    ax[0, 0].set_ylabel("AUROC", fontsize=7)
    ax[1, 0].set_ylabel("% of windows", fontsize=7)
    for j in range(1, 4):
        ax[1, j].set_ylabel("FP windows / h", fontsize=7)
    ax[0, 0].legend(fontsize=6, frameon=False)
    for a in ax.ravel():
        a.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=300)


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, "results_study"),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(base, "results_study", "sweep.png"))
