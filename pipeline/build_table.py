"""Player-week table from nflverse play-by-play, rosters and snaps, 2022-2026.
One row per offensive player per team game: what he did, and everything known before kickoff."""
import pandas as pd, numpy as np
import os
S = os.environ['WORK']; D = os.environ['DATA']
cols = ['game_id','season','week','season_type','posteam','defteam','home_team','away_team','spread_line','total_line','roof','wind','temp',
        'play_type','pass_attempt','rush_attempt','qb_dropback','sack','yardline_100','air_yards','yards_gained','complete_pass','interception',
        'pass_touchdown','rush_touchdown','touchdown','td_player_id','fumble_lost','fumbled_1_player_id','two_point_conv_result','two_point_attempt',
        'passer_player_id','receiver_player_id','rusher_player_id','epa','qb_scramble','home_score','away_score']
games, plays = [], []
for y in range(2022, 2027):
    f = f'{D}/play_by_play_{y}.parquet'
    p = pd.read_parquet(f, columns=cols); p = p[p.season_type == 'REG']
    plays.append(p)
p = pd.concat(plays, ignore_index=True)
# --- games: Vegas closing lines and weather
g = p.groupby('game_id').first()[['season','week','home_team','away_team','spread_line','total_line','roof','wind','temp','home_score','away_score']].reset_index()
tg = []
for _, r in g.iterrows():
    for team, opp, home in ((r.home_team, r.away_team, 1), (r.away_team, r.home_team, 0)):
        me = r.spread_line if home else -r.spread_line
        tg.append(dict(game_id=r.game_id, season=r.season, week=r.week, team=team, opp=opp, home=home, spread=me, total=r.total_line,
                       implied=r.total_line / 2 + me / 2, indoor=int(r.roof in ('dome', 'closed')), wind=r.wind, temp=r.temp,
                       pts=r.home_score if home else r.away_score))
tg = pd.DataFrame(tg)
# --- team volume per game
pp = p[p.play_type.isin(['pass', 'run'])]
tv = pp.groupby(['game_id', 'posteam']).agg(dropbacks=('qb_dropback', 'sum'), rushes=('rush_attempt', 'sum'), plays=('play_type', 'size'),
     targets=('receiver_player_id', lambda x: x.notna().sum()), rz_t=('receiver_player_id', lambda x: 0)).reset_index().rename(columns={'posteam': 'team'})
pp = pp.assign(rz=(pp.yardline_100 <= 20).astype(int), gl=(pp.yardline_100 <= 5).astype(int))
tv2 = pp.groupby(['game_id', 'posteam']).apply(lambda x: pd.Series({
    'tm_rz_targets': x.receiver_player_id.notna()[x.rz == 1].sum(), 'tm_rz_rush': ((x.rush_attempt == 1) & (x.rz == 1)).sum(),
    'tm_air': x.air_yards.where(x.receiver_player_id.notna()).sum(), 'pass_epa': x.epa[x.qb_dropback == 1].mean(), 'rush_epa': x.epa[x.rush_attempt == 1].mean()})).reset_index().rename(columns={'posteam': 'team'})
tv = tv.drop(columns='rz_t').merge(tv2, on=['game_id', 'team'])
# --- player usage per game
rec = pp[pp.receiver_player_id.notna()].groupby(['game_id', 'posteam', 'receiver_player_id']).agg(
    tgt=('play_type', 'size'), rec=('complete_pass', 'sum'), rec_yd=('yards_gained', lambda x: 0), air=('air_yards', 'sum'), rz_tgt=('rz', 'sum')).reset_index()
c = pp[pp.receiver_player_id.notna()]
rec['rec_yd'] = c.assign(y=c.yards_gained * c.complete_pass).groupby(['game_id', 'posteam', 'receiver_player_id']).y.sum().values
rec['rec_td'] = c.groupby(['game_id', 'posteam', 'receiver_player_id']).pass_touchdown.sum().values
rec = rec.rename(columns={'receiver_player_id': 'pid', 'posteam': 'team'})
ru = pp[pp.rusher_player_id.notna() & (pp.rush_attempt == 1)]
rush = ru.groupby(['game_id', 'posteam', 'rusher_player_id']).agg(car=('play_type', 'size'), rush_yd=('yards_gained', 'sum'), rush_td=('rush_touchdown', 'sum'),
    rz_car=('rz', 'sum'), gl_car=('gl', 'sum')).reset_index().rename(columns={'rusher_player_id': 'pid', 'posteam': 'team'})
pa = pp[pp.passer_player_id.notna()]
pas = pa.groupby(['game_id', 'posteam', 'passer_player_id']).agg(att=('pass_attempt', 'sum'), cmp=('complete_pass', 'sum'),
    pass_yd=('yards_gained', lambda x: 0), pass_td=('pass_touchdown', 'sum'), ints=('interception', 'sum'), qb_db=('qb_dropback', 'sum'), qb_epa=('epa', 'mean')).reset_index()
pas['pass_yd'] = pa.assign(y=pa.yards_gained * pa.complete_pass * pa.pass_attempt).groupby(['game_id', 'posteam', 'passer_player_id']).y.sum().values
pas = pas.rename(columns={'passer_player_id': 'pid', 'posteam': 'team'})
fl = p[p.fumble_lost == 1].groupby(['game_id', 'fumbled_1_player_id']).size().rename('fum').reset_index().rename(columns={'fumbled_1_player_id': 'pid'})
tp = p[(p.two_point_attempt == 1) & (p.two_point_conv_result == 'success')]
two = pd.concat([tp[['game_id', 'receiver_player_id']].rename(columns={'receiver_player_id': 'pid'}), tp[['game_id', 'rusher_player_id']].rename(columns={'rusher_player_id': 'pid'}),
                 tp[['game_id', 'passer_player_id']].rename(columns={'passer_player_id': 'pid'})]).dropna().groupby(['game_id', 'pid']).size().rename('two').reset_index()
pw = rec.merge(rush, on=['game_id', 'team', 'pid'], how='outer').merge(pas, on=['game_id', 'team', 'pid'], how='outer').merge(fl, on=['game_id', 'pid'], how='left').merge(two, on=['game_id', 'pid'], how='left').fillna(0)
# --- rosters: name, position, sleeper id, and who was inactive
ro = pd.concat([pd.read_parquet(f'{D}/roster_weekly_{y}.parquet', columns=['season', 'week', 'team', 'position', 'status', 'full_name', 'gsis_id', 'sleeper_id', 'pfr_id', 'game_type']) for y in range(2022, 2027)])
ro = ro[(ro.game_type == 'REG') & ro.position.isin(['QB', 'RB', 'WR', 'TE'])]
ro = ro.merge(tg[['season', 'week', 'team', 'game_id']], on=['season', 'week', 'team'])
pw = pw.merge(ro[['game_id', 'gsis_id', 'position', 'full_name', 'sleeper_id', 'pfr_id', 'status']].rename(columns={'gsis_id': 'pid'}), on=['game_id', 'pid'], how='left')
pw = pw[pw.position.isin(['QB', 'RB', 'WR', 'TE'])]
pw['ppr'] = pw.rec + .1 * pw.rec_yd + 6 * pw.rec_td + .1 * pw.rush_yd + 6 * pw.rush_td + .04 * pw.pass_yd + 4 * pw.pass_td - pw.ints - 2 * pw.fum + 2 * pw.two
pw = pw.merge(tg, on=['game_id', 'team']).merge(tv, on=['game_id', 'team'])
pw.to_parquet(f'{S}/player_games.parquet'); tg.merge(tv, on=['game_id', 'team']).to_parquet(f'{S}/team_games.parquet'); ro.to_parquet(f'{S}/rosters.parquet')
print(len(pw), pw.groupby('season').size().to_dict())
if False: print(pw[(pw.season == 2026) & (pw.full_name.isin(['Ladd McConkey', 'Rome Odunze']))][['week', 'full_name', 'tgt', 'rec', 'rec_yd', 'rec_td', 'ppr']].sort_values(['full_name', 'week']).to_string())
