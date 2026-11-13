"""How results spread around our projections, measured on games each model never saw.

For every stat, the model is refit with one season held out at a time; each held-out game gives
(projection, result). Games are grouped by position and projection level, and each group keeps
the spread of result / projection as 51 quantiles. The phone reads a bet's chance straight from
the group for that player's position and projection: zeros, blowups and skew included.
Writes dist.json. Run with CHECK=1 to also print a calibration table for 2026."""
import os, sys, json, re
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get('WORK', os.path.join(ROOT, 'work'))
src = open(os.path.join(ROOT, 'pipeline', 'run.py')).read()
F = eval(re.search(r"F = (\[.*?\])", src, re.S).group(1))
pw = pd.read_parquet(f'{WORK}/features.parquet'); pw = pw[pw.games >= 1].copy()
pw['pos_i'] = pw.position.map({'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3})
reg = lambda: HistGradientBoostingRegressor(learning_rate=.05, max_iter=300, min_samples_leaf=100, l2_regularization=1.0, random_state=1)
STATS = {'rec': 'rec', 'rec_yd': 'rec_yd', 'rush_yd': 'rush_yd', 'rush_att': 'car', 'pass_yd': 'pass_yd', 'pass_td': 'pass_td',
         'pass_int': 'ints', 'pass_att': 'att', 'pass_cmp': 'cmp'}
QB_ONLY = {'pass_yd', 'pass_td', 'pass_int', 'pass_att', 'pass_cmp'}
QS = np.linspace(0, 1, 51)
seasons = sorted(pw.season.unique())

def oof(col, d):
    out = pd.Series(np.nan, index=d.index)
    for s in seasons:
        tr, te = d[d.season != s], d[d.season == s]
        if len(te) == 0: continue
        out[te.index] = np.clip(reg().fit(tr[F], tr[col].fillna(0)).predict(te[F]), 0, None)
    return out

def groups(d, pred, actual):
    """Position x projection level: five levels by projection within each position."""
    res = []
    for pos, g in d.groupby('position'):
        p = pred[g.index]; a = actual[g.index].fillna(0)
        ok = p > 0.05
        p, a = p[ok], a[ok]
        if len(p) < 200: continue
        edges = np.unique(np.quantile(p, [0, .2, .4, .6, .8, 1]))
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p <= hi)
            if m.sum() < 100: continue
            res.append(dict(pos=pos, lo=float(lo), hi=float(hi), n=int(m.sum()), q=[round(float(x), 4) for x in np.quantile(a[m] / p[m], QS)]))
    return res

def chance(model, pos, proj, line):
    gs = [g for g in model if g['pos'] == pos] or model
    g = min(gs, key=lambda g: 0 if g['lo'] <= proj <= g['hi'] else min(abs(proj - g['lo']), abs(proj - g['hi'])))
    r = line / proj
    return 1 - np.interp(r, g['q'], QS, left=0, right=1)

out, preds = {}, {}
for name, col in STATS.items():
    d = pw[pw.att >= 10] if name in QB_ONLY else pw
    p = oof(col, d); preds[name] = (d, p)
    out[name] = groups(d[d.season < seasons[-1]] if os.environ.get('CHECK') else d, p, d[col])
    print(name, len(out[name]), 'groups', file=sys.stderr)
json.dump(out, open(os.path.join(ROOT, 'dist.json'), 'w'))

if os.environ.get('CHECK'):
    # Calibration on the latest season, using groups built only from earlier seasons: lines at the
    # projection and at 0.5x, 1.5x and 2x it (low lines and milestones), rounded to .5 like real lines.
    last = seasons[-1]
    rows = []
    for name, col in STATS.items():
        d, p = preds[name]
        t = d[(d.season == last) & (p > .05)]
        for mult in (.5, 1, 1.5, 2):
            for i in t.index:
                line = np.floor(p[i] * mult) + .5
                rows.append(dict(stat=name, mult=mult, pred=chance(out[name], t.position[i], p[i], line), hit=float(t[col].fillna(0)[i] > line)))
    r = pd.DataFrame(rows)
    r['band'] = pd.cut(r.pred, [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    print(r.groupby('band', observed=True).agg(said=('pred', 'mean'), happened=('hit', 'mean'), n=('hit', 'size')).round(3).to_string())
    print(r.groupby(['stat', 'mult']).agg(said=('pred', 'mean'), happened=('hit', 'mean'), n=('hit', 'size')).round(3).to_string())
