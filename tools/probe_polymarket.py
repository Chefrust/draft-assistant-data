"""Checks Polymarket's /markets filters for NFL bets with real money behind them."""
import json, urllib.request, collections, datetime
def get(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'draft-assistant', 'Accept': 'application/json'}), timeout=60) as r: return r.read()
now = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
types = ['anytime_touchdowns', 'two_plus_touchdowns', 'passing_yards', 'passing_touchdowns', 'passing_attempts', 'rushing_yards', 'receiving_yards', 'receptions', 'totals', 'spreads', 'moneyline']
base = 'https://gamma-api.polymarket.com/markets?tag_id=450&liquidity_num_min=1000&end_date_min=' + now + ''.join('&sports_market_types=' + t for t in types)
allm, size = [], 0
for lim in (100, 50):
    try:
        raw = get(base + f'&limit={lim}&offset=0'); print('limit', lim, 'ok', len(raw), 'bytes'); break
    except Exception as e: print('limit', lim, 'failed', e)
for off in range(0, 3000, lim):
    raw = get(base + f'&limit={lim}&offset={off}'); size += len(raw); page = json.loads(raw)
    allm += page
    if len(page) < lim: break
print('markets', len(allm), 'bytes', size)
c = collections.Counter(m.get('sportsMarketType') for m in allm); print(c.most_common())
for m in allm[:12]:
    print((m.get('question'), m.get('line'), m.get('outcomes'), m.get('bestBid'), m.get('bestAsk'), m.get('liquidity'), m.get('gameStartTime'), [e.get('slug') for e in m.get('events', [])][:1]))
liq = [m for m in allm if m.get('bestBid') is not None and m.get('bestAsk') is not None and float(m['bestAsk']) - float(m['bestBid']) <= .08]
print('tight', len(liq), collections.Counter(m.get('sportsMarketType') for m in liq).most_common())
