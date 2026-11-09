"""Pre-game features for every player-game: only information from earlier games, plus what's known
before kickoff (Vegas lines, weather, who's inactive, who starts at QB)."""
import pandas as pd, numpy as np
import os
S = os.environ['WORK']
import os
pw = pd.read_parquet(os.environ.get('PW', f'{S}/player_games.parquet')); tg = pd.read_parquet(os.environ.get('TG', f'{S}/team_games.parquet')); ro = pd.read_parquet(f'{S}/rosters.parquet')
OUT = os.environ.get('OUT', f'{S}/features.parquet')
pw['t'] = pw.season * 100 + pw.week
sn = pd.read_parquet(f'{S}/snaps.parquet').drop(columns='team')
pw = pw.merge(sn, on=['season', 'week', 'pfr_id'], how='left')
pw['snap'] = pw.offense_pct.fillna(pw.offense_pct.median())
tg['t'] = tg.season * 100 + tg.week
# shares of team volume
pw['tgt_sh'] = pw.tgt / pw.targets.clip(lower=1); pw['car_sh'] = pw.car / pw.rushes.clip(lower=1)
pw['rzt_sh'] = pw.rz_tgt / pw.tm_rz_targets.clip(lower=1); pw['rzc_sh'] = pw.rz_car / pw.tm_rz_rush.clip(lower=1)
pw['air_sh'] = pw.air / pw.tm_air.where(pw.tm_air > 0, 1)
pw['opp_n'] = pw.tgt + pw.car
pw = pw.sort_values(['pid', 't']).reset_index(drop=True)
def ew(col, hl, by='pid', w=None):
    s = pw.groupby(by)[col].transform(lambda x: x.shift().ewm(halflife=hl, min_periods=1).mean())
    return s
for c in ['tgt_sh', 'car_sh', 'snap']:
    pw[f'k_{c}'] = ew(c, 1)
for c in ['snap', 'tgt_sh', 'car_sh', 'rzt_sh', 'rzc_sh', 'air_sh', 'ppr', 'tgt', 'car', 'rec', 'rec_yd', 'rush_yd', 'rec_td', 'rush_td', 'att', 'pass_yd', 'pass_td', 'ints', 'qb_epa', 'gl_car']:
    pw[f'e_{c}'] = ew(c, 3)
    pw[f'l_{c}'] = ew(c, 12)
# efficiency, summed so far (ratio of sums), shrunk to position average
for num, den, name, k in [('rec_yd', 'tgt', 'ypt', 30), ('rec', 'tgt', 'catch', 30), ('rush_yd', 'car', 'ypc', 40), ('rec_td', 'tgt', 'tdpt', 60), ('rush_td', 'car', 'tdpc', 60)]:
    cn = pw.groupby('pid')[num].transform(lambda x: x.shift().ewm(halflife=8, min_periods=1).sum() if False else x.shift().rolling(24, min_periods=1).sum())
    cd = pw.groupby('pid')[den].transform(lambda x: x.shift().rolling(24, min_periods=1).sum())
    pos_rate = pw.groupby('position')[num].transform('sum') / pw.groupby('position')[den].transform('sum').clip(lower=1)
    pw[name] = (cn.fillna(0) + k * pos_rate) / (cd.fillna(0) + k)
pw['games'] = pw.groupby('pid').cumcount()
pw['games_season'] = pw.groupby(['pid', 'season']).cumcount()
# first game back: missed the team's previous game but played earlier this season
team_weeks = tg.groupby(['season', 'team']).week.apply(sorted).to_dict()
def prev_week(r):
    ws = team_weeks.get((r.season, r.team), []); i = ws.index(r.week) if r.week in ws else -1
    return ws[i - 1] if i > 0 else None
pw['prev_tw'] = pw.apply(prev_week, axis=1)
played = set(zip(pw.pid, pw.season, pw.week))
pw['missed_last'] = [int(pv is not None and (pid, s, pv) not in played) for pid, s, pv in zip(pw.pid, pw.season, pw.prev_tw)]
pw['first_back'] = ((pw.missed_last == 1) & (pw.games_season > 0)).astype(int)
# teammates out this week: share held by regulars who are not active
act = ro[['season', 'week', 'team', 'gsis_id', 'status']].rename(columns={'gsis_id': 'pid'})
last = pw[['pid', 'team', 'season', 'week', 'e_tgt_sh', 'e_car_sh', 'position']].copy()
# each player's latest share going into each team week
rows = []
for (s, team), x in pw.groupby(['season', 'team']):
    for wk in sorted(x.week.unique()):
        before = x[x.week < wk]
        if before.empty: continue
        recent = before[before.week >= wk - 4]
        sh = recent.groupby('pid').agg(tsh=('tgt_sh', 'mean'), csh=('car_sh', 'mean'))
        now = set(x[x.week == wk].pid)
        out = sh[~sh.index.isin(now)]
        rows.append(dict(season=s, team=team, week=wk, vac_tgt=out.tsh.sum(), vac_car=out.csh.sum()))
vac = pd.DataFrame(rows)
pw = pw.merge(vac, on=['season', 'team', 'week'], how='left').fillna({'vac_tgt': 0, 'vac_car': 0})
# this game's QB and his quality going in
# Upcoming games have no passes yet: the starter is the QB marked in the upcoming rows.
pw.loc[pw.get('starter', pd.Series(False, index=pw.index)).fillna(False).astype(bool), 'att'] = 1e-6
qbg = pw[pw.att > 0].sort_values('att').groupby(['game_id', 'team']).tail(1)[['game_id', 'team', 'pid', 'e_qb_epa', 'l_qb_epa', 'games']].rename(columns={'pid': 'qb', 'e_qb_epa': 'qb_e', 'l_qb_epa': 'qb_l', 'games': 'qb_games'})
pw = pw.merge(qbg, on=['game_id', 'team'], how='left')
# usual QB for the team over the last 4 games vs this game's QB
usual = qbg.merge(tg[['game_id', 'team', 't']], on=['game_id', 'team']).sort_values('t')
usual['usual_qb'] = usual.groupby('team').qb.transform(lambda x: x.shift().rolling(4, min_periods=1).apply(lambda v: pd.Series(v).mode().iloc[0], raw=False) if False else x.shift())
pw = pw.merge(usual[['game_id', 'team', 'usual_qb']], on=['game_id', 'team'], how='left')
pw['qb_change'] = (pw.qb != pw.usual_qb).astype(int)
# team tendencies going in
tg = tg.sort_values(['team', 't'])
tg['pass_rate'] = tg.dropbacks / tg.plays.clip(lower=1)
for c in ['pass_rate', 'plays', 'pass_epa', 'rush_epa', 'pts']:
    tg[f'te_{c}'] = tg.groupby('team')[c].transform(lambda x: x.shift().ewm(halflife=4, min_periods=1).mean())
# defense: what opponents did against them, by position (fantasy points) and per play
dpos = pw.groupby(['game_id', 'opp', 'position']).ppr.sum().unstack(fill_value=0).reset_index().rename(columns={'opp': 'def'})
dteam = tg[['game_id', 'team', 'opp', 't', 'pass_epa', 'rush_epa', 'pts']].rename(columns={'opp': 'def', 'team': 'off'})
dd = dteam.merge(dpos, on=['game_id', 'def']).sort_values(['def', 't'])
for c in ['QB', 'RB', 'WR', 'TE', 'pass_epa', 'rush_epa', 'pts']:
    dd[f'd_{c}'] = dd.groupby('def')[c].transform(lambda x: x.shift().ewm(halflife=5, min_periods=1).mean())
pw = pw.merge(tg[['game_id', 'team', 'te_pass_rate', 'te_plays', 'te_pass_epa', 'te_rush_epa', 'te_pts']], on=['game_id', 'team'], how='left')
pw = pw.merge(dd[['game_id', 'def', 'd_QB', 'd_RB', 'd_WR', 'd_TE', 'd_pass_epa', 'd_rush_epa', 'd_pts']].rename(columns={'def': 'opp'}), on=['game_id', 'opp'], how='left')
pw['d_pos'] = np.select([pw.position == p for p in ['QB', 'RB', 'WR', 'TE']], [pw.d_QB, pw.d_RB, pw.d_WR, pw.d_TE])
pw['wind'] = np.where(pw.indoor == 1, 0, pw.wind.fillna(0)); pw['temp'] = np.where(pw.indoor == 1, 70, pw.temp.fillna(60))
pw.to_parquet(OUT)
print(pw.shape)
if False: print(pw[(pw.season == 2026) & (pw.full_name.isin(['Ladd McConkey']))][['week', 'e_tgt_sh', 'ypt', 'implied', 'qb_e', 'vac_tgt', 'first_back', 'd_pos', 'ppr']].round(3).to_string())
