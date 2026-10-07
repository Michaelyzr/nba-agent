"""Betting-language presentation, reference-price comparisons and parlay planning."""
import itertools
import math

from agents.market_analysis import walk_asks, quote_problems, _finite
from forecast.betting import probability
from data_sources.inplay_types import utc
import pandas as pd

TITLES = {'moneyline': 'Moneyline', 'spread': 'Spread', 'total': 'Total points'}


def default_contracts(model):
    margin = model['margin_mean'] if model else 0
    # Half-point lines avoid presenting a push as a loss.
    handicap = -(math.floor(margin)+.5)
    total = math.floor(model['total_mean'] if model else 220)+.5
    return [{'id': 'model-moneyline', 'kind': 'moneyline', 'line': None, 'quotes': {}},
            {'id': 'model-spread', 'kind': 'spread', 'line': handicap, 'quotes': {}},
            {'id': 'model-total', 'kind': 'total', 'line': total, 'quotes': {}}]


def make_board(game, snapshot, model, contracts, demo=False):
    contracts = list(contracts or [])
    kinds = {c['kind'] for c in contracts}
    contracts.extend(c for c in default_contracts(model) if c['kind'] not in kinds)
    rows = []
    for c in contracts:
        sides = ('over', 'under') if c['kind'] == 'total' else ('home', 'away')
        quotes = c.get('quotes', {})
        pair = [quotes.get(side) for side in sides]
        pair_ok = all(isinstance(q, dict) for q in pair) and pair[0].get('condition_id') == pair[1].get('condition_id') and pair[0].get('token_id') != pair[1].get('token_id')
        for side in sides:
            q = quotes.get(side) or {}
            # Reuse identity/time/fee gates after verifying the exact contract upstream.
            check = {**q, 'kind': 'moneyline'}
            errors = quote_problems(check, snapshot, side) if q else ['Reference unavailable']
            current = bool(q) and pair_ok and not errors and snapshot.get('quote_state') != 'final'
            displayable = bool(q) and pair_ok and not quote_problems(check, snapshot, side, max_age=float('inf'))
            if c['kind'] == 'total':
                label = side.title()+' '+f"{c['line']:g}"
            else:
                label = getattr(game, side+'_team')
                if c['kind'] == 'spread':
                    line = c['line'] if side == 'home' else -c['line']
                    label += f' {line:+g}'
                else:
                    label += ' to win'
            p = probability(model, c['kind'], side, c['line'])
            rows.append({'id': str(game.game_id)+':'+str(c['id'])+':'+side, 'contract_id': str(c['id']),
                         'game_id': str(game.game_id), 'match': game.away_team+' @ '+game.home_team,
                         'kind': c['kind'], 'market': TITLES[c['kind']], 'line': c['line'], 'side': side, 'label': label,
                         'probability': p, 'model_odds': 1/p if p and p > 1e-8 else None,
                         'reference_quote': q if displayable else None, 'reference_current': current,
                         'reference_status': 'Simulated reference' if demo and current else 'Current' if current else 'Last seen' if displayable else 'No matching reference',
                         'research_only': True, 'phase': snapshot.get('quote_state'), 'simulation': demo,
                         'as_of': snapshot['as_of'], 'model_as_of': snapshot.get('model_as_of', snapshot['as_of'])})
    return {'game_id': str(game.game_id), 'bets': rows, 'own_model': model, 'as_of': snapshot['as_of']}


def evaluate(row, stake=20):
    stake = _finite(stake, 'stake', lo=1, hi=100000)
    row = dict(row)
    q, p = row.get('reference_quote'), row.get('probability')
    if not row.get('simulation'):
        now = pd.Timestamp.now(tz='UTC')
        try:
            ages = [(now-utc(value)).total_seconds() for value in [row['model_as_of'], (q or {})['observed_at'], (q or {})['updated_at']]]
            if any(not math.isfinite(age) or age < 0 or age > 15 for age in ages):
                row['reference_current'] = False
        except (KeyError, ValueError, TypeError):
            row['reference_current'] = False
    fill = None
    if q:
        try:
            fill = walk_asks(q, stake, .002)
            if not fill['shares'] or fill['cost'] < float(q.get('minimum_notional', 0)) or fill['shares'] < float(q.get('minimum_shares', 0)):
                fill = None
        except (ValueError, KeyError, TypeError):
            pass
    odds = 1/fill['effective_price'] if fill else None
    usable = bool(fill) and row.get('reference_current', False) and p is not None
    ev = fill['shares']*p-fill['cost'] if usable else None
    robust = fill['shares']*max(0,p-.02)-fill['cost'] if usable else None
    return {**row, 'stake': stake, 'reference_odds': odds, 'expected_profit': ev,
            'stress_profit': robust, 'edge_pp': (p-1/odds)*100 if usable else None,
            'estimated_return': fill['shares'] if usable else None, 'cost': fill['cost'] if fill else None,
            'unused_stake': fill['unused_budget'] if fill else stake,
            'eligible': bool(usable and robust > 0 and (p-.02-1/odds) >= .01)}


def recommend_single(rows, stake=20, goal='profit'):
    evaluated = [evaluate(r, stake) for r in rows]
    options = [r for r in evaluated if r['eligible']]
    if goal not in ('profit', 'chance'):
        raise ValueError('Unknown recommendation goal')
    if not options:
        return {'pick': None, 'reason': 'No positive value after costs and a model stress buffer. Wait or review a different market.'}
    metric = 'stress_profit' if goal == 'profit' else 'probability'
    return {'pick': max(options, key=lambda r: (r[metric], r['probability'] if goal == 'profit' else r['stress_profit'])),
            'reason': 'Highest estimated profit among the available positive-value choices.' if goal == 'profit' else 'Highest model win chance among the available positive-value choices.'}


def parlay_result(rows, stake=20):
    stake = _finite(stake, 'stake', lo=1, hi=100000)
    if not 2 <= len(rows) <= 4 or len({r['game_id'] for r in rows}) != len(rows):
        raise ValueError('Choose two to four picks, one per game')
    # Product is a planning benchmark, not a live combined/executable quote.
    picks = [evaluate(r, stake) for r in rows]
    if any(not r['reference_current'] or not r['reference_odds'] or r['probability'] is None for r in picks):
        return None
    p = math.prod(r['probability'] for r in picks)
    odds = math.prod(r['reference_odds'] for r in picks)
    stress_p = math.prod(max(0, r['probability']-.02) for r in picks)
    return {'picks': picks, 'probability': p, 'combined_odds': odds, 'model_odds': 1/p if p else None,
            'estimated_return': stake*odds, 'expected_profit': stake*(p*odds-1),
            'stress_profit': stake*(stress_p*odds-1), 'stake': stake,
            'assumption': 'Different games assumed independent. Return is a reference-odds benchmark; confirm an actual combined quote.',
            'research_only': True, 'executable_quote': False}


def recommend_parlay(rows, stake=20, legs=2, goal='profit'):
    if legs not in (2, 3, 4) or goal not in ('profit', 'chance'):
        raise ValueError('Choose 2–4 legs and a supported goal')
    groups = {}
    for r in rows:
        e = evaluate(r, stake)
        if e['eligible']:
            groups.setdefault(r['game_id'], []).append(r)
    if len(groups) > 12:
        raise ValueError('Select at most twelve games')
    # Exhaustive over the available positive-value picks, one outcome per distinct game.
    best = None
    for ids in itertools.combinations(groups, legs):
        for chosen in itertools.product(*(groups[i] for i in ids)):
            result = parlay_result(chosen, stake)
            if result is None or result['stress_profit'] <= 0:
                continue
            key = (result['stress_profit'], result['probability']) if goal == 'profit' else (result['probability'], result['stress_profit'])
            if best is None or key > best[0]:
                best = (key, result)
    return {'parlay': best[1] if best else None,
            'reason': 'Best estimated profit within the selected games, bet types and leg count.' if best and goal == 'profit' else 'Highest estimated win chance within your selected options.' if best else 'Not enough distinct games with positive-value, current references. Reduce the leg count or wait.'}


def simulated_contracts(game, model, snapshot, previous_model=None, lines_model=None):
    """Explicitly fake, past-only market lag; never used to generate model probabilities."""
    contracts = default_contracts(lines_model or previous_model or model)
    prior = previous_model or model
    for i, c in enumerate(contracts):
        c['id'] = 'sim-'+c['kind']+(':'+str(c['line']) if c['line'] is not None else '')
        c['quotes'] = {}
        for side in (('over','under') if c['kind']=='total' else ('home','away')):
            p = probability(prior, c['kind'], side, c['line'])
            if p is None or snapshot.get('quote_state') == 'final':
                continue
            mid = min(.97,max(.03,p))
            # Small hypothetical pricing differences illustrate both buy and wait paths.
            shift = min(.07, p*.20, (1-p)*.20)
            mid = min(.97,max(.03,mid + (-shift if side in ('away','over') else shift)))
            ask = min(.995,mid+.012)
            c['quotes'][side] = {'game_id': str(game.game_id), 'side': side, 'kind': 'moneyline',
                'condition_id': 'sim-'+str(game.game_id)+'-'+c['id'], 'token_id': 'sim-'+str(game.game_id)+'-'+c['id']+'-'+side,
                'includes_overtime': True, 'rules_verified': True, 'active': True, 'synthetic': True,
                'fee_rate': .05, 'fee_exponent': 1, 'fee_verified': True, 'minimum_notional': 1,
                'bid': max(.005,mid-.012), 'ask': ask, 'mid': mid,
                'asks': [{'price': ask, 'size': 100}, {'price': min(.999,ask+.015), 'size': 500}],
                'bids': [{'price': max(.005,mid-.012), 'size': 100}, {'price': max(.001,mid-.027), 'size': 500}],
                'observed_at': snapshot['as_of'], 'updated_at': snapshot['as_of']}
    return contracts


def hedge_bet(rows, pick_id, original_stake, placed_odds, budget=20):
    amount = _finite(original_stake, 'original_stake', lo=.01)
    odds = _finite(placed_odds, 'placed_odds', lo=1.000001)
    budget = _finite(budget, 'budget', lo=1, hi=100000)
    payout = _finite(amount*odds, 'payout', lo=.01)
    row = next((r for r in rows if r['id'] == pick_id), None)
    other = next((r for r in rows if row and r['game_id'] == row['game_id'] and r['contract_id'] == row['contract_id'] and r['side'] != row['side']), None)
    if not other or not evaluate(other, budget)['reference_current']:
        raise ValueError('A current price for the exact opposite bet is required')
    q = other['reference_quote']
    f = walk_asks(q, budget, .002, max_shares=payout)
    if not f['shares'] or f['cost'] < float(q.get('minimum_notional', 0)) or f['shares'] < float(q.get('minimum_shares', 0)):
        raise ValueError('Not enough reference depth for this hedge')
    a, b = payout-amount-f['cost'], f['shares']-amount-f['cost']
    return {'label': other['label'], 'amount': f['cost'], 'floor': min(a,b), 'before': -amount,
            'if_original_wins': a, 'if_opposite_wins': b, 'balanced': abs(f['shares']-payout) < 1e-6}
