"""Builds feature rows for one upcoming week: active players on teams playing that week.
usage: predict.py SEASON WEEK"""
import sys, os, subprocess, pickle, pandas as pd, numpy as np
S = os.environ['WORK']; D = os.environ['DATA']
season, week = int(sys.argv[1]), int(sys.argv[2])
pw = pd.read_parquet(f'{S}/player_games.parquet'); tg = pd.read_parquet(f'{S}/team_games.parquet'); ro = pd.read_parquet(f'{S}/rosters.parquet')
g = pd.read_csv(f'{D}/games.csv'); g = g[(g.season == season) & (g.week == week) & (g.game_type == 'REG')]
# drop anything already played in that week so the projection only uses earlier games
pw = pw[~((pw.season == season) & (pw.week == week))]; tg = tg[~((tg.season == season) & (tg.week == week))]
rows_t = []
for _, r in g.iterrows():
    for team, opp, home in ((r.home_team, r.away_team, 1), (r.away_team, r.home_team, 0)):
        me = r.spread_line if home else -r.spread_line
        rows_t.append(dict(game_id=r.game_id, season=season, week=week, team=team, opp=opp, home=home, spread=me, total=r.total_line,
                           implied=r.total_line / 2 + me / 2, indoor=int(r.roof in ('dome', 'closed')), wind=r.wind, temp=r.temp))
nt = pd.DataFrame(rows_t)
# roster for the week (latest available), active skill players only
# straight from the roster file: the merged table only has teams whose game is already played
rw = pd.read_parquet(f'{D}/roster_weekly_{season}.parquet')
rw = rw[(rw.game_type == 'REG') & rw.position.isin(['QB', 'RB', 'WR', 'TE'])]
use = week if week in set(rw.week) else rw.week.max()
act = rw[(rw.week == use) & (rw.status == 'ACT')][['team', 'gsis_id', 'position', 'full_name', 'sleeper_id', 'pfr_id']].rename(columns={'gsis_id': 'pid'})
out_ids = set(os.environ.get('OUT_IDS', '').split(',')) - {''}
act = act[~act.pid.isin(out_ids)]
# only players who have played before
seen = set(pw.pid)
act = act[act.pid.isin(seen)]
nr = act.merge(nt, on='team')
# starting QB: last game's starter if active, else the active QB with the most past attempts
last_qb = pw[pw.att > 0].sort_values(['season', 'week', 'att']).groupby('team').tail(1).set_index('team').pid.to_dict()
att = pw.groupby('pid').att.sum().to_dict()
def starter(x, team):
    qbs = x[x.position == 'QB']
    if qbs.empty: return x.assign(starter=False)
    lq = last_qb.get(team)
    pick = lq if lq in set(qbs.pid) else max(qbs.pid, key=lambda p: att.get(p, 0))
    return x.assign(starter=x.pid == pick)
nr = pd.concat([starter(x, t) for t, x in nr.groupby('team')], ignore_index=True)
pw2 = pd.concat([pw.assign(starter=False), nr], ignore_index=True)
tg2 = pd.concat([tg, nt], ignore_index=True)
pw2.to_parquet(f'{S}/_pw.parquet'); tg2.to_parquet(f'{S}/_tg.parquet')
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features.py')], env={**os.environ, 'PW': f'{S}/_pw.parquet', 'TG': f'{S}/_tg.parquet', 'OUT': f'{S}/_feat.parquet'}, check=True)
