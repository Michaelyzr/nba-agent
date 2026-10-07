"""English customer view and real position hedging; read-only, no trade execution."""
import math

from agents.market_analysis import compare_routes, quote_problems, walk_asks, _finite
from agents.game_timeline import clock_label, elapsed_seconds

EVENT_LABELS = {"left_injured": "left with an injury", "injury_out": "ruled out for the game",
                "returned": "returned to the game", "ejected": "ejected", "fouled_out": "fouled out",
                "questionable_return": "return uncertain", "doubtful_return": "unlikely to return",
                "rescinded": "ejection rescinded", "review": "update under review"}


def sell_position(quote, shares, extra=.002):
    shares = _finite(shares, 'shares')
    extra = _finite(extra, 'extra', hi=.5)
    rate = _finite(quote['fee_rate'], 'fee_rate', hi=1)
    exponent = _finite(quote.get('fee_exponent', 1), 'fee_exponent', lo=.01, hi=5)
    remaining, proceeds, fees, filled = shares, 0., 0., 0.
    levels = []
    for row in quote.get('bids', []):
        p, n = float(row['price']), float(row['size'])
        if not math.isfinite(n) or n < 0 or not 0 < p < 1:
            raise ValueError('invalid bid levels')
        levels.append((p, n))
    for p, n in sorted(levels, reverse=True):
        n = min(n, remaining)
        fee = rate*(p*(1-p))**exponent
        proceeds += n*(p-fee-extra); fees += n*fee; filled += n; remaining -= n
        if remaining <= 1e-8:
            break
    if filled < float(quote.get('minimum_shares', 0)) or filled and proceeds < float(quote.get('minimum_notional', 0)):
        return {'shares': 0., 'proceeds': 0., 'fees': 0.}
    return {'shares': filled, 'proceeds': proceeds, 'fees': fees}


def hedge_options(snapshot, quotes, position, budget=100, extra=.002):
    budget = _finite(budget, 'budget', hi=100000)
    extra = _finite(extra, 'extra', hi=.5)
    if not position:
        return []
    side, shares, original_cost = position['side'], float(position['shares']), float(position['cost'])
    if side not in ('home', 'away') or not all(math.isfinite(v) for v in (shares, original_cost)) or shares <= 0 or not 0 <= original_cost <= shares:
        raise ValueError('position must have a side, positive shares and cost between zero and payout')
    own, opposite = quotes.get(side), quotes.get('away' if side == 'home' else 'home')
    base = {'kind': 'hold', 'side': side, 'title': 'Hold your position', 'added_cost': 0., 'shares': 0.,
            'if_held_wins': shares-original_cost, 'if_other_wins': -original_cost, 'floor': -original_cost}
    options = [base]
    if snapshot.get('quote_state') == 'final':
        return options
    pair_ok = isinstance(own, dict) and isinstance(opposite, dict) and own.get('condition_id') == opposite.get('condition_id') and own.get('token_id') != opposite.get('token_id')
    if not pair_ok:
        return options
    if not quote_problems(opposite, snapshot, 'away' if side == 'home' else 'home'):
        try:
            fill = walk_asks(opposite, budget, extra, max_shares=shares)
            n, cost = fill['shares'], fill['cost']
            minimum = cost >= float(opposite.get('minimum_notional', 0)) and n >= float(opposite.get('minimum_shares', 0))
            if n and minimum:
                a, b = shares-original_cost-cost, n-original_cost-cost
                options.append({'kind': 'hedge', 'side': 'away' if side == 'home' else 'home', 'title': 'Buy the opposite outcome',
                                'added_cost': cost, 'shares': n, 'if_held_wins': a, 'if_other_wins': b, 'floor': min(a, b),
                                'balanced': abs(n-shares) < 1e-6, 'fees': fill['fees']})
        except (ValueError, TypeError, KeyError):
            pass
    if not quote_problems(own, snapshot, side):
        try:
            sold = sell_position(own, shares, extra)
            if sold['shares']:
                remaining = shares-sold['shares']
                a, b = remaining+sold['proceeds']-original_cost, sold['proceeds']-original_cost
                options.append({'kind': 'reduce', 'side': side, 'title': 'Reduce your position', 'added_cost': 0.,
                                'shares': sold['shares'], 'proceeds': sold['proceeds'], 'fees': sold['fees'],
                                'if_held_wins': a, 'if_other_wins': b, 'floor': min(a,b)})
        except (ValueError, TypeError, KeyError):
            pass
    return options


def next_action(snapshot, quotes, home, away, budget=100, position=None, demo=False):
    names = {'home': home, 'away': away}
    if snapshot.get('quote_state') == 'final':
        return {'title': 'Game finished', 'kind': 'finished', 'reason': 'The final score is confirmed. No new game-winner action.', 'amount': 0, 'options': []}
    if position:
        options = hedge_options(snapshot, quotes, position, budget)
        best = max(options, key=lambda r: (r['floor'], -r['added_cost']))
        label = 'Hold your position'
        reason = 'No available hedge improves the lower of the two game-result payouts.'
        if best['kind'] == 'hedge':
            label = f"Hedge with {names[best['side']]}"
            reason = f"Buy {best['shares']:.1f} opposite shares. Lower game-result P/L improves from ${options[0]['floor']:.2f} to ${best['floor']:.2f}."
        elif best['kind'] == 'reduce':
            label = f"Reduce {names[best['side']]}"
            reason = f"Selling {best['shares']:.1f} shares improves the lower game-result P/L to ${best['floor']:.2f}."
        return {**best, 'title': label, 'reason': reason, 'amount': best['added_cost'], 'options': options,
                'simulation': demo, 'note': 'Two completed-game results; canceled-game payouts depend on the market rules. Recheck fills before trading.'}
    result = compare_routes(snapshot, quotes, budget, uncertainty_pp=2)
    candidate = result['candidate'] if demo else result['decision']
    if candidate == 'wait':
        reason = 'No clear advantage after costs. Keep your budget available.'
        if not demo and result['candidate'] != 'wait' and result['decision_reasons']:
            reason = 'A price difference exists, but the model or live data is not ready for a buy signal.'
        if not quotes:
            reason = 'Waiting for a matching, current Polymarket market.'
        elif snapshot.get('p_home') is None:
            reason = ('Waiting for tip-off. Live model odds start with current game data.'
                      if snapshot.get('quote_state') == 'waiting_for_tip'
                      else 'Waiting for current game data before comparing model odds.')
        elif all(r['problems'] for r in result['routes']):
            reason = 'Market prices are unavailable, stale or not matched to this game.'
        return {'title': 'Wait for a better entry', 'kind': 'wait', 'reason': reason, 'amount': 0, 'options': [], 'comparison': result}
    row = next(r for r in result['routes'] if r['side'] == candidate)
    return {'title': f"Buy {names[candidate]}", 'kind': 'buy', 'side': candidate, 'amount': row['fill']['cost'],
            'shares': row['fill']['shares'], 'ev': row['ev'], 'edge_pp': row['edge_pp'],
            'reason': f"The estimated win chance is {row['edge_pp']:.1f} percentage points above the all-in entry cost.",
            'simulation': demo, 'options': [], 'comparison': result}


def event_feed(steps):
    out = []
    seen = set()
    for step in steps:
        s = step['snapshot']
        for e in s.get('report', {}).get('new_evidence', []):
            effect = s['report'].get('news_effect') or {}
            if e['id'] in seen:
                continue
            seen.add(e['id'])
            out.append({'id': e['id'], 'headline': e['player']+' '+EVENT_LABELS.get(e['status'], 'status updated'),
                        'clock': clock_label(s['score']) if s.get('score') else '', 'as_of': s['as_of'],
                        'impact_pp': effect.get('home_delta_pp'), 'selected': e['selected'],
                        'source': e['source'], 'synthetic': e['synthetic'], 'url': e.get('url', ''), 'text': e['text'],
                        'published_at': e.get('published_at'), 'observed_at': e.get('observed_at')})
        for item in s.get('retrieved_news', []):
            ident = item['item_id']
            if ident in seen or any(e.get('url') and e.get('url') == item.get('url') for e in s.get('report', {}).get('new_evidence', [])):
                continue
            seen.add(ident)
            out.append({'id': ident, 'headline': item.get('title') or item.get('text') or 'Match news update',
                        'clock': clock_label(s['score']) if s.get('score') else '', 'as_of': s['as_of'],
                        'impact_pp': None, 'selected': False, 'source': item['source'], 'synthetic': bool(item.get('synthetic')),
                        'url': item.get('url', ''), 'text': item.get('text', ''),
                        'published_at': item.get('published_at'), 'observed_at': item.get('observed_at'), 'news_only': True})
    return out


def match_view(payload, steps, snapshot, quotes=None, demo=True, budget=100, position=None):
    budget = _finite(budget, 'budget', lo=1, hi=100000)
    quotes = snapshot.get('market_quotes', {}) if quotes is None else quotes
    comparison = compare_routes(snapshot, quotes, budget, uncertainty_pp=2)
    teams = []
    for side in ('home', 'away'):
        row = next(r for r in comparison['routes'] if r['side'] == side)
        q, fill = quotes.get(side) or {}, row['fill'] or {}
        book_ok = bool(q) and not quote_problems(q, snapshot, side)
        book_matches = bool(q) and not quote_problems(q, snapshot, side, max_age=float('inf'))
        other = quotes.get('away' if side == 'home' else 'home')
        if other and (q.get('condition_id') != other.get('condition_id') or q.get('token_id') == other.get('token_id')):
            book_ok = False
            book_matches = False
        if book_matches and not fill:
            try:
                fill = walk_asks(q, budget, .002)
            except (ValueError, TypeError, KeyError):
                book_ok = False
                book_matches = False
        displayable = book_matches and bool(fill.get('shares')) and fill.get('cost', 0) >= float(q.get('minimum_notional', 0)) and fill.get('shares', 0) >= float(q.get('minimum_shares', 0))
        book_ok = book_ok and displayable
        p = row['p_model']
        teams.append({'side': side, 'team': payload[side+'_team'], 'probability': p,
                      'model_odds': 1/p if p else None, 'market_odds': 1/fill['effective_price'] if displayable else None,
                      'market_ask_odds': 1/q['ask'] if q.get('ask') else None,
                      'entry_price': fill.get('effective_price'), 'edge_pp': row['edge_pp'],
                      'quote_available': bool(book_ok), 'price_status': 'Current' if book_ok else 'Last seen' if displayable else 'Unavailable'})
    curve = []
    for step in steps:
        s = step['snapshot']
        if not s.get('score') or s.get('quote_state') == 'final' or s.get('p_home') is None:
            continue
        q = s.get('market_quotes', {}).get('home') or {}
        curve.append({'minute': elapsed_seconds(s['score'])/60, 'probability': s['p_home'], 'market': q.get('mid') if q and not quote_problems(q, s, 'home') else None,
                      'clock': clock_label(s['score'])})
    return {'game_id': snapshot['game_id'], 'home_team': payload['home_team'], 'away_team': payload['away_team'],
            'score': snapshot.get('score'), 'clock': clock_label(snapshot['score']) if snapshot.get('score') else 'Before tip-off',
            'quote_state': snapshot.get('quote_state'), 'as_of': snapshot['as_of'], 'simulation': demo,
            'news_health': snapshot.get('news_health'), 'freshness': snapshot.get('freshness'),
            'teams': teams, 'action': next_action(snapshot, quotes, payload['home_team'], payload['away_team'], budget, position, demo),
            'events': event_feed(steps), 'curve': curve, 'model_trained': snapshot.get('model_trained', False),
            'market_url': (quotes.get('home') or {}).get('url'), 'market_connected': bool(quotes),
            'market_updated_at': (quotes.get('home') or {}).get('observed_at'),
            'rules': (quotes.get('home') or {}).get('rules', 'Simulated full-game winner market, including overtime.' if demo else '')}
