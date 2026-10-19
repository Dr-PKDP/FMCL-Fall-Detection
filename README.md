# Federated fall detection on four public accelerometer datasets

## Data

Download the archives into one folder (`raw/`). Do not unzip them.

| Dataset | Source | Expected archive |
|---|---|---|
| UniMiB SHAR | http://www.sal.disco.unimib.it/technologies/unimib-shar/ | file name contains `unimib` |
| UMAFall | https://figshare.com/articles/dataset/UMA_ADL_FALL_Dataset_zip/4214283 | file name contains `4214283` or `umafall` |
| FallAllD | https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living | file name contains `fallalld` |
| KFall | https://sites.google.com/view/kfalldataset (access follows an approved registration form; download `sensor_data.zip`) | file name contains `kfall` or `sensor_data` |

## Run

```
pip install -r requirements.txt
python prepare_data.py --raw raw --out cache
bash run_study.sh 8
```

`prepare_data.py` writes `cache/cache_<dataset>.npz` and `cache/data_flow.json` (fall trials found, loaded and excluded per dataset) and `cache/file_manifest.csv` (every archive member read, with its role). `run_study.sh` runs 30 study jobs (six sets, five seeds), three calibration jobs and three balanced-aggregation jobs, then writes the tables, statistics and figure to `results_study/`. The argument is the number of parallel single-thread jobs. Set `FMCL_CACHE` if the cache is not in `cache/`.

Single jobs:

```
python study.py --site unimib --seed 0
python study.py --site fallalld --dev 2 --seed 0
python calibration.py --site umafall --rounds 15
python balanced_aggregation.py --site kfall --seed 0
python analyze_study.py results_study
python stats_tests.py results_study
python plot_sweep.py results_study results_study/sweep.png
python audit.py results_study
```

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | converts the archives to 50 Hz, 3 s trigger-centred windows |
| `common.py` | data loading, folds, model, federated averaging, centralised training, local-only distance |
| `study.py` | federated, centralised, pool-only, cross-site and local-only arms; leave-subjects-out with validation-fixed thresholds |
| `calibration.py` | per-user score calibration against the raw score, with sensitivity, false-positive windows and AUROC |
| `balanced_aggregation.py` | FEDALL with dataset-balanced client weights (KFall target) |
| `analyze_study.py` | merges job outputs; AUROC, achieved sensitivity, false-positive windows per hour, bootstrap intervals |
| `stats_tests.py` | non-inferiority and difference tests, Holm correction, random-effects pooling |
| `costs.py` | parameter count, bytes per round, multiply-accumulates, raw-window size |
| `plot_sweep.py` | figure for the rounds sweep |
| `audit.py` | per-subject recording time and the fall-trial data-flow table |
| `run_study.sh` | runs everything |
