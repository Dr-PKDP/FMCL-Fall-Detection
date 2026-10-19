import argparse
import io
import os
import re
import zipfile
from collections import defaultdict

import numpy as np

FS = 50
FS_IN = 100  # KFall inertial sensor sampling rate
WIN = 150
HALF = WIN // 2
TRIGGER_G = 1.5
RAW_DIR = "."
OUT_DIR = "cache"
FLOW = {}


def find_zip(*needles):
    for f in sorted(os.listdir(RAW_DIR)):
        if f.lower().endswith(".zip") and any(n.lower() in f.lower() for n in needles):
            return os.path.join(RAW_DIR, f)
    return None


FALL_MIN_TASK = 22
MANIFEST = []   # (dataset, archive member, fall or adl)


def resample_to_50(x, fs):
    x = np.asarray(x, dtype=np.float64)
    k = max(1, int(round(fs / FS)))
    if k > 1:
        ker = np.ones(k) / k
        x = np.stack([np.convolve(x[:, c], ker, mode="same") for c in range(x.shape[1])], axis=1)
    n = x.shape[0]
    t_old = np.arange(n) / fs
    t_new = np.arange(0, t_old[-1], 1.0 / FS)
    return np.stack([np.interp(t_new, t_old, x[:, c]) for c in range(x.shape[1])], axis=1)


def peak_windows(acc, thr=TRIGGER_G, take_max_only=False):
    from scipy.signal import find_peaks
    mag = np.linalg.norm(acc, axis=1)
    if len(mag) < 10:
        return []
    if take_max_only:
        idx = [int(np.argmax(mag))]
    else:
        idx, _ = find_peaks(mag, height=thr, distance=FS)
    out = []
    pad = np.pad(acc, ((HALF, HALF), (0, 0)), mode="edge")
    for i in idx:
        w = pad[i:i + WIN]                                
        out.append((w.T.astype(np.float32), float(mag[i])))
    return out


def flow_entry(name, **kw):
    f = FLOW.setdefault(name, {})
    for k, v in kw.items():
        f[k] = f.get(k, 0) + v


def finish_flow(name, rows):
    f = FLOW.setdefault(name, {})
    f["fall_windows_built"] = sum(1 for r in rows if r["y"] == 1)
    f["fall_windows_below_trigger"] = sum(1 for r in rows if r["y"] == 1
                                          and np.linalg.norm(r["X"].astype(np.float64), axis=0).max() < TRIGGER_G)
    f["fall_windows_final"] = f["fall_windows_built"] - f["fall_windows_below_trigger"]
    f["adl_windows_built"] = sum(1 for r in rows if r["y"] == 0)


def save(name, rows, exp, fall_n, extra=None):
    finish_flow(name, rows)
    X = np.stack([r["X"] for r in rows]).astype(np.float16)
    d = dict(
        X=X,
        y=np.array([r["y"] for r in rows], np.uint8),
        subj=np.array([r["subj"] for r in rows], np.int16),
        trial=np.array([r["trial"] for r in rows], np.int32),
        act=np.array([r["act"] for r in rows], np.int16),
        age=np.array([r["age"] for r in rows], np.int16),
        dev=np.array([r["dev"] for r in rows], np.int8),
        peak=np.array([r["peak"] for r in rows], np.float32),
        exp_key=np.array(sorted(exp), np.int32),
        exp_hours=np.array([exp[k] for k in sorted(exp)], np.float64),
        fall_trials_key=np.array(sorted(fall_n), np.int32),
        fall_trials_n=np.array([fall_n[k] for k in sorted(fall_n)], np.int32),
    )
    if extra:
        d.update(extra)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"cache_{name}.npz")
    np.savez_compressed(path, **d)
    print(f"  saved {path}  ({os.path.getsize(path)/1e6:.1f} MB)  windows={len(rows)}  "
      f"ADL={int((d['y']==0).sum())}  fall={int(d['y'].sum())}  subjects={len(set(d['subj'].tolist()))}")


def prep_unimib():
    print("UniMiB SHAR")
    import scipy.io as sio
    path = find_zip("unimib")
    if not path:
        print("  zip not found")
        return
    zf = zipfile.ZipFile(path)

    def load(suffix):
        n = next(i for i in zf.namelist() if i.endswith(suffix))
        return sio.loadmat(io.BytesIO(zf.read(n)))

    data = load("data/acc_data.mat")["acc_data"]
    lab = load("data/acc_labels.mat")["acc_labels"].astype(int)
    full = load("data/full_data.mat")["full_data"]
    N = data.shape[0]
    print(f"  windows {data.shape}  labels {lab.shape}")

    a = data.reshape(N, 3, 151)
    b = data.reshape(N, 151, 3).transpose(0, 2, 1)
    sm = lambda w: float(np.mean(np.abs(np.diff(w, axis=2))))
    pick_a = sm(a[:500]) <= sm(b[:500])
    print(f"  layout check: block-wise roughness={sm(a[:500]):.3f}  interleaved roughness={sm(b[:500]):.3f}"
      f"  -> using {'block-wise' if pick_a else 'interleaved'}")
    W = (a if pick_a else b)[:, :, :WIN].astype(np.float64)

    med = float(np.median(np.median(np.linalg.norm(W, axis=1), axis=1)))
    scale = 1.0 / 9.80665 if med > 4.0 else 1.0
    print(f"  median window magnitude {med:.2f} -> scale {scale:.4f} ({'m/s^2 -> g' if scale != 1 else 'already g'})")
    W = W * scale

    ages, genders = {}, {}
    for r in range(full.shape[0]):
        ages[r + 1] = int(np.asarray(full[r, 2]).ravel()[0])
        genders[r + 1] = str(np.asarray(full[r, 1]).ravel()[0]).strip()
    adl_fields = ["Walking", "Running", "GoingUpS", "GoingDownS", "Jumping", "SittingDown",
                  "StandingUpFS", "LyingDownFS", "StandingUpFL"]
    fall_fields = ["FallingForw", "FallingBack", "FallingRight", "FallingLeft", "Syncope",
                   "FallingBackSC", "FallingWithPS", "HittingObstacle"]
    exp, fall_n = {}, {}
    for r in range(full.shape[0]):
        rec = full[r, 0][0, 0]
        secs = 0.0
        for f in adl_fields:
            cell = rec[f]
            for j in range(cell.shape[0]):
                secs += cell[j, 0].shape[1] / FS
        exp[(r + 1) * 10] = secs / 3600.0
        nf = 0
        for f in fall_fields:
            nf += rec[f].shape[0]
        fall_n[(r + 1) * 10] = nf
    print(f"  recorded ADL hours total {sum(exp.values()):.2f}; fall recordings {sum(fall_n.values())}")

    rows = []
    for i in range(N):
        act, subj, trial = int(lab[i, 0]), int(lab[i, 1]), int(lab[i, 2])
        rows.append(dict(X=W[i], y=int(act >= 10), subj=subj, trial=act * 100 + trial, act=act,
                         age=ages.get(subj, -1), dev=0, peak=float(np.linalg.norm(W[i][:, HALF]))))
    print("  ages (subject:age):", {k: v for k, v in sorted(ages.items())})
    flow_entry("unimib", fall_trials_found=int(sum(1 for r in rows if r["y"] == 1)),
               fall_trials_loaded=int(sum(1 for r in rows if r["y"] == 1)), fall_excluded=0)
    save("unimib", rows, exp, fall_n)


def prep_umafall():
    print("UMAFall (phone in right pocket, accelerometer)")
    path = find_zip("4214283", "umafall")
    if not path:
        print("  zip not found")
        return
    outer = zipfile.ZipFile(path)
    pick = next(n for n in outer.namelist() if "corrected" in n.lower() and n.lower().endswith(".zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(pick)))
    rows, exp, fall_n = [], defaultdict(float), defaultdict(int)
    act_codes, ages_seen = {}, {}
    tcount = defaultdict(int)
    skipped = 0
    infos = [i for i in inner.infolist() if i.filename.lower().endswith(".csv")]
    for n, info in enumerate(infos):
        base = os.path.basename(info.filename)
        m = re.match(r"UMAFall_Subject_(\d+)_(ADL|Fall)_(.+?)_(\d+)_\d{4}-", base)
        if not m:
            continue
        subj, kind, act, trial = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
        MANIFEST.append(("umafall", info.filename, kind.lower()))
        if kind.lower() == "fall":
            flow_entry("umafall", fall_trials_found=1)
        txt = inner.read(info).decode("utf-8", errors="replace")
        age = -1
        xyz, ts = [], []
        for ln in txt.splitlines():
            s = ln.strip()
            if not s:
                continue
            if s.startswith("%"):
                mm = re.match(r"%\s*Age\s*:\s*(\d+)", s)
                if mm:
                    age = int(mm.group(1))
                continue
            c = s.split(";")
            if len(c) >= 7 and c[5].strip() == "0" and c[6].strip() == "0":
                try:
                    ts.append(float(c[0]))
                    xyz.append((float(c[2]), float(c[3]), float(c[4])))
                except ValueError:
                    pass
        if len(xyz) < 200:
            skipped += 1
            if kind.lower() == "fall":
                flow_entry("umafall", fall_excluded=1)
            continue
        if kind.lower() == "fall":
            flow_entry("umafall", fall_trials_loaded=1)
        ages_seen[subj] = age
        xyz = np.array(xyz)
        dur = (ts[-1] - ts[0]) / 1000.0
        fs = (len(xyz) - 1) / dur if dur > 1 else 200.0
        acc = resample_to_50(xyz, fs)
        key = subj * 10
        is_fall = kind.lower() == "fall"
        code = act_codes.setdefault(act, len(act_codes) + 1)
        tcount[subj] += 1
        if is_fall:
            fall_n[key] += 1
            wins = peak_windows(acc, take_max_only=True)
        else:
            exp[key] += len(acc) / FS / 3600.0
            wins = peak_windows(acc)
        for w, pk in wins:
            rows.append(dict(X=w, y=int(is_fall), subj=subj, trial=tcount[subj], act=code,
                             age=age, dev=0, peak=pk))
        if (n + 1) % 150 == 0:
            print(f"    ... {n+1}/{len(infos)} files")
    print(f"  files read {len(infos)-skipped}, skipped {skipped}")
    print("  subject ages:", dict(sorted(ages_seen.items())))
    print("  activity codes:", act_codes)
    mags = np.array([np.median(np.linalg.norm(r["X"].astype(np.float64), axis=0)) for r in rows[:400]])
    print(f"  sanity: median magnitude across first windows = {np.median(mags):.2f} g (expect near 1)")
    save("umafall", rows, exp, fall_n)


def prep_fallalld():
    print("FallAllD (accelerometer, three devices)")
    path = find_zip("fallalld")
    if not path:
        print("  zip not found")
        return
    zf = zipfile.ZipFile(path)
    rx = re.compile(r"S(\d+)_D(\d+)_A(\d+)_T(\d+)_A\.dat$")
    infos = []
    for i in zf.infolist():
        m = rx.search(i.filename)
        if m:
            infos.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), i))
    print(f"  accelerometer files {len(infos)}")
    SCALE = 1.0 / 4096.0
    rows, exp, fall_n = [], defaultdict(float), defaultdict(int)
    rest_mags = []
    for n, (s, d, a, t, info) in enumerate(infos):
        raw = zf.read(info).decode("ascii", errors="ignore").strip().replace("\r", "")
        try:
            arr = np.fromstring(raw.replace("\n", ","), sep=",")
            arr = arr.reshape(-1, 3)
        except Exception:
            arr = np.loadtxt(io.StringIO(raw), delimiter=",")
        g = arr * SCALE
        acc = resample_to_50(g, 238.0)
        is_fall = a >= 100
        MANIFEST.append(("fallalld", info.filename, "fall" if is_fall else "adl"))
        if is_fall:
            flow_entry("fallalld", fall_trials_found=1, fall_trials_loaded=1)
        key = s * 10 + d
        if len(rest_mags) < 300 and not is_fall:
            rest_mags.append(float(np.median(np.linalg.norm(acc, axis=1))))
        if is_fall:
            fall_n[key] += 1
            wins = peak_windows(acc, take_max_only=True)
        else:
            exp[key] += len(acc) / FS / 3600.0
            wins = peak_windows(acc)
        for w, pk in wins:
            rows.append(dict(X=w, y=int(is_fall), subj=s, trial=a * 100 + t, act=a, age=-1,
                             dev=d, peak=pk))
        if (n + 1) % 1000 == 0:
            print(f"    ... {n+1}/{len(infos)} files")
    print(f"  sanity: median resting magnitude over {len(rest_mags)} ADL files = "
      f"{np.median(rest_mags):.2f} g (expect near 1; if not, the 1/4096 scale is wrong)")
    save("fallalld", rows, exp, fall_n)


def prep_kfall():
    path = find_zip("kfall", "sensor_data")
    if not path:
        print("no KFall zip found in the raw folder")
        return
    print("using", path)
    zf = zipfile.ZipFile(path)
    rx = re.compile(r"S(\d+)T(\d+)R(\d+)\.csv$", re.I)
    infos = []
    for i in zf.infolist():
        m = rx.search(i.filename)
        if m and "__MACOSX" not in i.filename:
            infos.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), i))
    print(f"sensor files: {len(infos)}")
    rows, exp, fall_n = [], defaultdict(float), defaultdict(int)
    task_peaks = defaultdict(list)
    task_trig = defaultdict(list)
    rest = []
    bad = 0
    for n, (s, t, r, info) in enumerate(infos):
        MANIFEST.append(("kfall", info.filename, "fall" if t >= FALL_MIN_TASK else "adl"))
        if t >= FALL_MIN_TASK:
            flow_entry("kfall", fall_trials_found=1)
        try:
            arr = np.genfromtxt(io.BytesIO(zf.read(info)), delimiter=",", skip_header=1, usecols=(2, 3, 4))
        except Exception:
            bad += 1
            if t >= FALL_MIN_TASK:
                flow_entry("kfall", fall_excluded=1)
            continue
        arr = arr[~np.isnan(arr).any(axis=1)]
        if len(arr) < 50:
            bad += 1
            if t >= FALL_MIN_TASK:
                flow_entry("kfall", fall_excluded=1)
            continue
        acc = resample_to_50(arr, FS_IN)
        mag = np.linalg.norm(acc, axis=1)
        is_fall = t >= FALL_MIN_TASK
        if is_fall:
            flow_entry("kfall", fall_trials_loaded=1)
        task_peaks[t].append(float(mag.max()))
        if t == 1 and len(rest) < 40:
            rest.append(float(np.median(mag)))
        if is_fall:
            fall_n[s * 10] += 1
            wins = peak_windows(acc, take_max_only=True)
        else:
            exp[s * 10] += len(acc) / FS / 3600.0
            wins = peak_windows(acc)
            task_trig[t].append(len(wins))
        for w, pk in wins:
            rows.append(dict(X=w, y=int(is_fall), subj=s, trial=t * 100 + r, act=t, age=-1, dev=0, peak=pk))
        if (n + 1) % 500 == 0:
            print(f"   ... {n+1}/{len(infos)} files")
    print(f"unreadable files skipped: {bad}")
    if rest:
        print(f"sanity: median resting magnitude in task 1 = {np.median(rest):.3f} g (expect about 1)")
    print("\nper task: files, median / 10th / 90th percentile of the largest magnitude (g); for ADL tasks the mean "
      "number of >=1.5 g triggers per file")
    for t in sorted(task_peaks):
        v = np.array(task_peaks[t])
        trig = f"  triggers/file {np.mean(task_trig[t]):.2f}" if t in task_trig else ""
        print(f"   T{t:02d} {'FALL' if t >= FALL_MIN_TASK else 'ADL '}  n={len(v):4d}  "
          f"median {np.median(v):5.2f}  p10 {np.percentile(v,10):5.2f}  p90 {np.percentile(v,90):5.2f}{trig}")
    adl_p = np.concatenate([task_peaks[t] for t in task_peaks if t < FALL_MIN_TASK])
    fall_p = np.concatenate([task_peaks[t] for t in task_peaks if t >= FALL_MIN_TASK])
    print(f"\nlargest magnitude, ADL files: median {np.median(adl_p):.2f} g;  fall files: median {np.median(fall_p):.2f} g")

    finish_flow("kfall", rows)
    X = np.stack([r["X"] for r in rows]).astype(np.float16)
    d = dict(
        X=X, y=np.array([r["y"] for r in rows], np.uint8), subj=np.array([r["subj"] for r in rows], np.int16),
        trial=np.array([r["trial"] for r in rows], np.int32), act=np.array([r["act"] for r in rows], np.int16),
        age=np.array([r["age"] for r in rows], np.int16), dev=np.array([r["dev"] for r in rows], np.int8),
        peak=np.array([r["peak"] for r in rows], np.float32),
        exp_key=np.array(sorted(exp), np.int32), exp_hours=np.array([exp[k] for k in sorted(exp)], np.float64),
        fall_trials_key=np.array(sorted(fall_n), np.int32), fall_trials_n=np.array([fall_n[k] for k in sorted(fall_n)], np.int32))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "cache_kfall.npz")
    np.savez_compressed(out, **d)
    print(f"\nsaved {out} ({os.path.getsize(out)/1e6:.1f} MB): windows {len(rows)}, ADL {int((d['y']==0).sum())}, "
      f"fall {int(d['y'].sum())}, subjects {len(set(d['subj'].tolist()))}, ADL hours {sum(exp.values()):.2f}")


SITES = {"unimib": prep_unimib, "umafall": prep_umafall, "fallalld": prep_fallalld, "kfall": prep_kfall}


def main():
    global RAW_DIR, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=".", help="folder holding the downloaded zip archives")
    ap.add_argument("--out", default="cache", help="folder for the cache_*.npz files")
    ap.add_argument("--sites", nargs="*", default=list(SITES))
    a = ap.parse_args()
    RAW_DIR, OUT_DIR = a.raw, a.out
    for s in a.sites:
        SITES[s]()
    import json
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "data_flow.json"), "w") as f:
        json.dump(FLOW, f, indent=1)
    with open(os.path.join(OUT_DIR, "file_manifest.csv"), "w") as f:
        f.write("dataset,archive_member,role\n")
        for row in MANIFEST:
            f.write(",".join(row) + "\n")


if __name__ == "__main__":
    main()
