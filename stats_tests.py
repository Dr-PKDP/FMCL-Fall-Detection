import json, sys, numpy as np
from scipy import stats
rng = np.random.default_rng(2024)
import os
IN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_study')
SETS = ['unimib','umafall','fallalld_D1','fallalld_D2','fallalld_D3','kfall']
ARMS = ['FED','CEN','LOC','FEDALL']
B = 5000; DELTA = 0.02

def load(tag):
    rows = json.load(open(f'{IN}/rows_{tag}_r60.json'))
    subj = {}
    for r in rows:
        d = subj.setdefault(r['subj'], {'exp_h': r['exp_h'], 'n_adl': r['n_adl'], **{f'auc_{a}': [] for a in ARMS}, **{f'fp_{a}': [] for a in ARMS}})
        for a in ARMS:
            if r[f'auc_{a}'] is not None: d[f'auc_{a}'].append(r[f'auc_{a}'])
            d[f'fp_{a}'].append(r[f'fp_{a}'])
    return subj

def auc_table(subj):
    ids = [s for s,d in subj.items() if all(len(d[f'auc_{a}']) > 0 for a in ARMS)]
    return {s: {a: float(np.mean(subj[s][f'auc_{a}'])) for a in ARMS} for s in ids}

def boot_mean(x):
    x = np.asarray(x); n = len(x)
    m = x[rng.integers(0, n, (B, n))].mean(1)
    return m

data = {t: load(t) for t in SETS}
auc = {t: auc_table(data[t]) for t in SETS}
comps = [('FED','CEN'), ('FED','LOC'), ('FEDALL','FED')]
out = {'delta': DELTA, 'per_dataset': {}, 'holm': [], 'meta': {}, 'false_alarms': {}}
pv = []
for t in SETS:
    ids = sorted(auc[t]); res = {'n_subjects': len(ids)}
    for a, b in comps:
        d = np.array([auc[t][s][a] - auc[t][s][b] for s in ids])
        bm = boot_mean(d)
        lo, hi = np.percentile(bm, [2.5, 97.5]); lb90 = np.percentile(bm, 5)
        r = {'mean': float(d.mean()), 'ci95': [float(lo), float(hi)], 'lower90': float(lb90), 'se': float(bm.std())}
        r['n_zero_diff'] = int((d == 0).sum()); r['n_tied_abs'] = int(len(d) - len(np.unique(np.abs(d))))
        if (a, b) == ('FED','CEN'):
            # non-inferiority: H0 median diff <= -delta
            w = stats.wilcoxon(d + DELTA, alternative='greater')
            r['p'] = float(w.pvalue); r['test'] = 'Wilcoxon one-sided, non-inferiority delta=0.02'
            r['noninferior_bootstrap'] = bool(lb90 > -DELTA)
        elif (a, b) == ('FED','LOC'):
            w = stats.wilcoxon(d, alternative='greater')
            r['p'] = float(w.pvalue); r['test'] = 'Wilcoxon one-sided, FED > LOC'
        else:
            w = stats.wilcoxon(d, alternative='two-sided')
            r['p'] = float(w.pvalue); r['test'] = 'Wilcoxon two-sided'
        res[f'{a}-{b}'] = r; pv.append((t, f'{a}-{b}', r['p']))
    out['per_dataset'][t] = res
# Holm
order = sorted(range(len(pv)), key=lambda i: pv[i][2]); m = len(pv); adj = [None]*m; run = 0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * pv[i][2])); adj[i] = run
for (t, c, p), q in zip(pv, adj):
    out['per_dataset'][t][c]['p_holm'] = float(q)
# meta-analysis over 4 independent datasets (FallAllD devices combined per subject id)
def effect(t_list, a, b):
    ids = set()
    for t in t_list: ids |= set(auc[t])
    d = []
    for s in sorted(ids):
        v = [auc[t][s][a] - auc[t][s][b] for t in t_list if s in auc[t]]
        d.append(np.mean(v))
    d = np.array(d); bm = boot_mean(d)
    return float(d.mean()), float(bm.var()), len(d)
def dl_hksj(y, v):
    y = np.array(y); v = np.array(v); k = len(y)
    w = 1 / v; ybar = (w*y).sum()/w.sum()
    Q = (w*(y-ybar)**2).sum(); C = w.sum() - (w**2).sum()/w.sum()
    tau2 = max(0.0, (Q-(k-1))/C)
    ws = 1/(v+tau2); mu = (ws*y).sum()/ws.sum()
    q = (ws*(y-mu)**2).sum()/(k-1)
    se = np.sqrt(q/ws.sum())
    tcrit = stats.t.ppf(0.975, k-1)
    I2 = max(0.0, (Q-(k-1))/Q) if Q > 0 else 0.0
    return {'mu': float(mu), 'ci95_hksj': [float(mu-tcrit*se), float(mu+tcrit*se)], 'tau2': float(tau2), 'I2': float(I2),
            'p_hksj': float(2*(1-stats.t.cdf(abs(mu/se), k-1))), 'k': k}
groups = {'UniMiB':['unimib'], 'UMAFall':['umafall'], 'FallAllD':['fallalld_D1','fallalld_D2','fallalld_D3'], 'KFall':['kfall']}
for a, b in comps:
    ys, vs, ns = [], [], []
    for g, ts in groups.items():
        y, v, n = effect(ts, a, b); ys.append(y); vs.append(v); ns.append(n)
    r = dl_hksj(ys, vs); r['per_dataset'] = dict(zip(groups, [round(y, 4) for y in ys]))
    out['meta'][f'{a}-{b}'] = r
# false-positive windows per hour of recorded activity (UniMiB: percent of activity windows)
def conc(ids, fp, n_adl, hours):
    order = sorted(ids, key=lambda s: (-fp[s], s)); q = int(np.ceil(0.25 * len(order))); top, rest = order[:q], order[q:]
    f = lambda g, v: float(sum(v[s] for s in g))
    res = {'n_users': len(order), 'n_top': q, 'fp_share': f(top, fp) / f(order, fp),
           'window_share': f(top, n_adl) / f(order, n_adl),
           'fp_fraction_top': f(top, fp) / f(top, n_adl), 'fp_fraction_rest': f(rest, fp) / f(rest, n_adl)}
    if hours is not None: res['time_share'] = f(top, hours) / f(order, hours)
    return res
out['concentration'] = {}
for t in SETS:
    pct = t == 'unimib'
    ids = [s for s in sorted(data[t]) if data[t][s]['n_adl'] >= 10 and (pct or data[t][s]['exp_h'] > 0)]; R = {'unit': 'percent of activity windows' if pct else 'per recorded hour'}
    den = np.array([data[t][s]['n_adl'] / 100 if pct else data[t][s]['exp_h'] for s in ids])
    rate = {a: np.array([np.mean(data[t][s][f'fp_{a}']) for s in ids])/den for a in ARMS}
    idx = rng.integers(0, len(ids), (B, len(ids)))
    for a in ARMS:
        bs = rate[a][idx].mean(1)
        R[a] = {'rate': float(rate[a].mean()), 'ci95': [float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5))]}
    dd = rate['FED'] - rate['CEN']; bd = dd[idx].mean(1)
    R['FED-CEN'] = {'diff': float(dd.mean()), 'ci95': [float(np.percentile(bd,2.5)), float(np.percentile(bd,97.5))]}
    out['false_alarms'][t] = R
    fp = {s: float(np.mean(data[t][s]['fp_FED'])) for s in ids}
    out['concentration'][t] = conc(ids, fp, {s: data[t][s]['n_adl'] for s in ids}, None if pct else {s: data[t][s]['exp_h'] for s in ids})
json.dump(out, open(f'{IN}/stats_results.json','w'), indent=1)
# print summary
for t in SETS:
    print(t, out['per_dataset'][t]['n_subjects'])
    for c in ['FED-CEN','FED-LOC','FEDALL-FED']:
        r = out['per_dataset'][t][c]
        print(f"  {c:11s} d={r['mean']:+.4f} CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] p={r['p']:.2e} holm={r['p_holm']:.2e} zeros={r['n_zero_diff']} ties={r['n_tied_abs']}" + (f" NI={r['noninferior_bootstrap']}" if 'noninferior_bootstrap' in r else ''))
for c, r in out['meta'].items(): print('META', c, r)
for t in SETS:
    R = out['false_alarms'][t]
    print(t, R['unit'], {a: (round(R[a]['rate'],2), [round(x,2) for x in R[a]['ci95']]) for a in ['FED','CEN','FEDALL']}, 'FED-CEN', round(R['FED-CEN']['diff'],2), [round(x,2) for x in R['FED-CEN']['ci95']])
    print('   top quartile', {k: round(v, 3) for k, v in out['concentration'][t].items()})
