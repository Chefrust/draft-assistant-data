"""Prints what Polymarket's public API lists for the NFL, to see market types and question wording."""
import json, urllib.request, collections
def get(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'draft-assistant', 'Accept': 'application/json'}), timeout=60) as r: return json.load(r)
try:
    get('https://gamma-api.polymarket.com/events?limit=1')
    print('plain events call ok')
except Exception as e:
    print('plain events call failed', e, getattr(e, 'headers', ''))
for q in ['events?tag_slug=nfl&limit=5', 'events?tag_slug=nfl&closed=false&limit=5', 'events?tag_slug=nfl&active=true&limit=5', 'sports']:
    try:
        r = get('https://gamma-api.polymarket.com/' + q)
        print('OK', q, json.dumps(r)[:1500] if q == 'sports' else len(r))
    except Exception as e:
        print('FAIL', q, e)
sports = get('https://gamma-api.polymarket.com/sports')
nfl = [x for x in sports if str(x.get('sport', '')).lower() == 'nfl']
print('nfl sport entry', nfl)
ev = []
for off in range(0, 600, 50):
    try: page = get(f'https://gamma-api.polymarket.com/events?series_id=12185&closed=false&limit=50&offset={off}')
    except Exception as e: print('page fail', off, e); break
    if not page: break
    ev += page
print('events', len(ev))
types = collections.Counter(); ex = collections.defaultdict(list)
for e in ev:
    for m in e.get('markets', []):
        t = m.get('sportsMarketType') or '-'
        types[t] += 1
        if t in ('moneyline',) or 'spread' in t or 'totals' in t or t == 'exact_margin': continue
        if str(m.get('gameStartTime') or '') < '2026-11-14': continue
        if len(ex[t]) < 4: ex[t].append((e.get('title'), e.get('slug'), m.get('question'), m.get('line'), m.get('outcomes'), m.get('outcomePrices'), m.get('gameStartTime'), m.get('groupItemTitle')))
print(types.most_common())
print('keys', sorted(ev[-1]['markets'][0].keys()))
for t, xs in ex.items():
    print('==', t)
    for x in xs: print('  ', x)
print('titles', [e.get('title') for e in ev[:40]])
