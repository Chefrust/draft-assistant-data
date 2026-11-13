"""Prints what Polymarket's public API lists for the NFL, to see market types and question wording."""
import json, urllib.request, collections
def get(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'draft-assistant', 'Accept': 'application/json'}), timeout=60) as r: return json.load(r)
try:
    get('https://gamma-api.polymarket.com/events?limit=1')
    print('plain events call ok')
except Exception as e:
    print('plain events call failed', e, getattr(e, 'headers', ''))
ev = []
for off in range(0, 1000, 100):
    page = get(f'https://gamma-api.polymarket.com/events?tag_slug=nfl&closed=false&limit=100&offset={off}')
    if not page: break
    ev += page
print('events', len(ev))
types = collections.Counter(); ex = collections.defaultdict(list)
for e in ev:
    for m in e.get('markets', []):
        t = m.get('sportsMarketType') or '-'
        types[t] += 1
        if len(ex[t]) < 6: ex[t].append((e.get('title'), e.get('slug'), m.get('question'), m.get('line'), m.get('outcomes'), m.get('outcomePrices'), m.get('gameStartTime'), m.get('groupItemTitle')))
print(types.most_common())
for t, xs in ex.items():
    print('==', t)
    for x in xs: print('  ', x)
print('titles', [e.get('title') for e in ev[:40]])
