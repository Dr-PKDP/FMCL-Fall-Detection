#!/bin/bash
# usage: bash run_study.sh [parallel_jobs]
# 30 study jobs (6 datasets x 5 seeds), 3 calibration jobs and 3 balanced-aggregation jobs, then all tables and statistics.
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
CACHE="${FMCL_CACHE:-cache}"
N=${1:-8}

for s in unimib umafall fallalld kfall; do
  [ -f "$CACHE/cache_$s.npz" ] || { echo "missing $CACHE/cache_$s.npz (run prepare_data.py first)"; exit 1; }
done
python3 -c "import torch, sklearn, scipy, matplotlib" || { echo "missing python packages (pip install -r requirements.txt)"; exit 1; }
for f in study.py analyze_study.py calibration.py balanced_aggregation.py stats_tests.py common.py; do
  [ -f "$f" ] || { echo "missing $f in $(pwd)"; exit 1; }
done

mkdir -p logs results_study results_balanced
: > jobs_study.txt
for seed in 0 1 2 3 4; do
  for spec in "unimib" "umafall" "fallalld --dev 1" "fallalld --dev 2" "fallalld --dev 3" "kfall"; do
    echo "study.py --site $spec --seed $seed --save-scores" >> jobs_study.txt
  done
done
for spec in "unimib" "umafall" "fallalld --dev 1"; do
  echo "calibration.py --site $spec --rounds 15 --seeds 5 --out results_study" >> jobs_study.txt
done
for seed in 0 1 2; do
  echo "balanced_aggregation.py --site kfall --seed $seed --out results_balanced" >> jobs_study.txt
done

cat jobs_study.txt | xargs -P "$N" -I{} bash -c 'a="{}"; n=$(echo "$a" | tr -c "[:alnum:]" "_"); python3 $a > logs/$n.log 2>&1; echo "finished: $a"'

n=$(ls results_study/study_*_s[0-9].json 2>/dev/null | wc -l)
[ "$n" -eq 30 ] || { echo "only $n of 30 study files were written; check logs/"; exit 1; }
n=$(ls results_study/calibration_*.json 2>/dev/null | wc -l)
[ "$n" -eq 3 ] || { echo "only $n of 3 calibration files were written; check logs/"; exit 1; }
n=$(ls results_balanced/balanced_kfall_s[0-9].json 2>/dev/null | wc -l)
[ "$n" -eq 3 ] || { echo "only $n of 3 balanced-aggregation files were written; check logs/"; exit 1; }

python3 analyze_study.py results_study | tee results_study/study_report.txt
python3 stats_tests.py results_study | tee results_study/stats_report.txt
python3 balanced_aggregation.py --summary results_study results_balanced | tee results_balanced/balanced_report.txt
python3 plot_sweep.py results_study results_study/sweep.png
python3 costs.py | tee results_study/costs.txt
echo "all done"
