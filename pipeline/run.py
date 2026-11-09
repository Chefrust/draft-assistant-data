"""Daily: download public nflverse data, rebuild the tables, retrain, and project the next week.
Writes projections.csv at the repo root. Public data only."""
import os, subprocess, sys, urllib.request, pickle, datetime
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'work', 'data'); WORK = os.path.join(ROOT, 'work')
os.makedirs(DATA, exist_ok=True)
os.environ.update(DATA=DATA, WORK=WORK)
NV = 'https://github.com/nflverse/nflverse-data/releases/download'
season = int(os.environ.get('SEASON', '2026'))
years = range(2018, season + 1)
os.environ['FIRST'] = '2018'; os.environ['LAST'] = str(season)

def get(url, path, always=False):
    if os.path.exists(path) and not always: return
    with urllib.request.urlopen(url, timeout=300) as r, open(path, 'wb') as f: f.write(r.read())

for y in years:
    cur = y == season
    get(f'{NV}/pbp/play_by_play_{y}.parquet', f'{DATA}/play_by_play_{y}.parquet', cur)
    get(f'{NV}/weekly_rosters/roster_weekly_{y}.parquet', f'{DATA}/roster_weekly_{y}.parquet', cur)
    get(f'{NV}/snap_counts/snap_counts_{y}.parquet', f'{DATA}/snap_counts_{y}.parquet', cur)
for y in years:
    get(f'{NV}/injuries/injuries_{y}.parquet', f'{DATA}/injuries_{y}.parquet', y == season)
get('https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv', f'{DATA}/games.csv', True)

sn = pd.concat([pd.read_parquet(f'{DATA}/snap_counts_{y}.parquet') for y in years])
sn[sn.game_type == 'REG'][['season', 'week', 'pfr_player_id', 'team', 'offense_pct', 'offense_snaps']].rename(columns={'pfr_player_id': 'pfr_id'}).to_parquet(f'{WORK}/snaps.parquet')
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, f'{here}/build_table.py'], check=True)
subprocess.run([sys.executable, f'{here}/features.py'], check=True)

# --- train on every finished game
pw = pd.read_parquet(f'{WORK}/features.parquet'); pw = pw[pw.games >= 1].copy()
pw['pos_i'] = pw.position.map({'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3})
F = ['pos_i', 'e_snap', 'l_snap', 'k_snap', 'k_tgt_sh', 'k_car_sh', 'e_tgt_sh', 'l_tgt_sh', 'e_car_sh', 'l_car_sh', 'e_rzt_sh', 'e_rzc_sh', 'e_air_sh',
     'e_ppr', 'l_ppr', 'e_tgt', 'e_car', 'e_gl_car', 'e_att', 'l_att', 'e_pass_yd', 'e_pass_td', 'e_ints', 'e_qb_epa', 'l_qb_epa', 'ypt', 'catch', 'ypc',
     'tdpt', 'tdpc', 'games', 'games_season', 'first_back', 'missed_last', 'vac_tgt', 'vac_car', 'qb_e', 'qb_l', 'qb_games', 'qb_change', 'implied',
     'spread', 'total', 'home', 'indoor', 'wind', 'temp', 'te_pass_rate', 'te_plays', 'te_pass_epa', 'te_rush_epa', 'd_pos', 'd_pass_epa', 'd_rush_epa', 'd_pts']
reg = lambda: HistGradientBoostingRegressor(learning_rate=.05, max_iter=300, min_samples_leaf=100, l2_regularization=1.0, random_state=1)
models = {'ppr': reg().fit(pw[F], pw.ppr)}
for col in ['rec', 'rec_yd', 'rush_yd', 'rec_td', 'rush_td']:
    models[col] = reg().fit(pw[F], pw[col])
qb = pw[pw.att >= 10]
for col in ['pass_yd', 'pass_td', 'ints']:
    models[col] = reg().fit(qb[F], qb[col])

# --- next week: first week of this season with a game not yet played
g = pd.read_csv(f'{DATA}/games.csv')
g = g[(g.season == season) & (g.game_type == 'REG')]
open_weeks = g[g.home_score.isna()].week
if open_weeks.empty: print('season over'); sys.exit(0)
week = int(open_weeks.min())
inj = pd.read_parquet(f'{DATA}/injuries_{season}.parquet')
out_ids = set(inj[(inj.week == week) & (inj.report_status.isin(['Out']))].gsis_id)
os.environ['OUT_IDS'] = ','.join(out_ids)
subprocess.run([sys.executable, f'{here}/predict.py', str(season), str(week)], check=True)
f = pd.read_parquet(f'{WORK}/_feat.parquet'); f = f[(f.season == season) & (f.week == week)].copy()
f['pos_i'] = f.position.map({'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3})
f = f[(f.position != 'QB') | f.starter.fillna(False).astype(bool)]
for k, m in models.items():
    f[k] = np.clip(m.predict(f[F]), 0, None)
f.loc[f.position != 'QB', ['pass_yd', 'pass_td', 'ints']] = 0

def why(r):
    w = []
    if r.e_snap > 0: w.append(f'Snap share {r.e_snap * 100:.0f}%')
    if r.position in ('WR', 'TE') or (r.position == 'RB' and r.e_tgt_sh >= .08): w.append(f'Target share {r.e_tgt_sh * 100:.0f}%')
    if r.position == 'RB': w.append(f'Carry share {r.e_car_sh * 100:.0f}%')
    w.append(f'Team total {r.implied:.1f}')
    if pd.notna(r.qb_e) and r.position != 'QB':
        if r.qb_e <= -.1: w.append('QB playing poorly')
        elif r.qb_e >= .1: w.append('QB playing well')
    if r.qb_change == 1 and r.position != 'QB': w.append('Different QB this week')
    if r.vac_tgt >= .08 and r.position != 'QB': w.append(f'{r.vac_tgt * 100:.0f}% of targets open with teammates out')
    if r.vac_car >= .15 and r.position == 'RB': w.append(f'{r.vac_car * 100:.0f}% of carries open')
    if r.first_back == 1 and r.position != 'QB': w.append('First game back')
    if r.position == 'QB' and r.games_season <= 1: w.append('Few starts this season')
    return ' | '.join(w)
f['why'] = f.apply(why, axis=1)
# Banged up: shown as information. From 2021-2026, only a Questionable tag cost points against our
# projection (-11%); practice limits, recent listings and snap drops did not, once usage is in the model.
inj_all = inj[inj.game_type == 'REG'] if 'game_type' in inj.columns else inj
# each player's snap share in his last two games against his usual this season
fa = pd.read_parquet(f'{WORK}/_feat.parquet')
fa = fa[(fa.season == season) & (fa.week < week) & fa.offense_pct.notna()].sort_values('week')
snap_hist = {}
for pid_, x in fa.groupby('pid'):
    if len(x) >= 4:
        snap_hist[pid_] = (list(x.offense_pct.iloc[-2:]), float(x.offense_pct.iloc[:-2].median()))
inj_all = inj_all[~inj_all.report_primary_injury.fillna('').str.contains('Not injury related') & ~inj_all.practice_primary_injury.fillna('').str.contains('Not injury related')]
def banged(r):
    """Only the signs worth a look: limited or no practice or a designation this week, an injury
    listing last week or the week before, or a big snap drop last game. Full practice alone isn't flagged."""
    clean = lambda v: '' if v is None or str(v).strip().lower() in ('none', 'nan', '') else str(v).strip()
    notes = []
    now = inj_all[(inj_all.gsis_id == r.pid) & (inj_all.week == week)]
    if len(now):
        x = now.iloc[-1]
        part = clean(x.practice_status); desig = clean(x.report_status)
        body = clean(x.report_primary_injury) or clean(x.practice_primary_injury)
        if desig or (part and not part.startswith('Full')):
            bits = [body, part.replace(' in Practice', '').replace('Participation', 'participation').lower(), desig]
            notes.append('This week: ' + ', '.join(b for b in bits if b))
    past = inj_all[(inj_all.gsis_id == r.pid) & (inj_all.week.between(week - 2, week - 1))]
    past = past[past.practice_status.fillna('').str.startswith(('Limited', 'Did Not')) | past.report_status.notna()]
    if len(past):
        body = clean(past.iloc[-1].report_primary_injury) or clean(past.iloc[-1].practice_primary_injury)
        notes.append(('%s, ' % body if body else '') + 'hurt ' + ' and '.join(f'Week {int(w)}' for w in sorted(past.week.unique())))
    sn = snap_hist.get(r.pid)
    if sn is not None:
        last2, usual = sn
        if usual >= .4 and min(last2) <= usual - .25:
            notes.append('Snaps ' + ' and '.join(f'{v * 100:.0f}%' for v in last2) + f' the last {len(last2)} games (usual {usual * 100:.0f}%)')
    return ' | '.join(notes)
f['banged'] = f.apply(banged, axis=1)
f['sleeper_id'] = f.sleeper_id.astype(str).str.replace(r'\.0$', '', regex=True)
out = f[['season', 'week', 'sleeper_id', 'pid', 'full_name', 'position', 'team', 'opp', 'ppr', 'rec', 'rec_yd', 'rush_yd', 'rec_td', 'rush_td', 'pass_yd', 'pass_td', 'ints', 'why', 'banged']]
out = out.rename(columns={'pid': 'gsis_id', 'full_name': 'name', 'ints': 'pass_int'})
out = out.sort_values('ppr', ascending=False).round(3)
out.to_csv(os.path.join(ROOT, 'projections.csv'), index=False)
with open(os.path.join(ROOT, 'updated.txt'), 'w') as fh:
    fh.write(f'{datetime.datetime.now(datetime.UTC):%Y-%m-%d %H:%M} UTC  season {season} week {week}  {len(out)} players  {len(out_ids)} ruled out\n')
print(open(os.path.join(ROOT, 'updated.txt')).read())
