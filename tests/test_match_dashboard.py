"""Customer actions, exact automatic market discovery and refresh isolation."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from agents.match_view import hedge_options, match_view, next_action, sell_position
from agents.dashboard_live import LiveDashboard, discover_games
from data_sources.polymarket_books import PolymarketReader, standard_rules
from test_market_analysis import state_and_quotes, market
from test_inplay import NOW


def test_hedge_balances_two_game_payouts_without_exceeding_position(tmp_path):
    s, q = state_and_quotes(tmp_path)
    position = {'side': 'home', 'shares': 100, 'cost': 60}
    options = hedge_options(s, q, position, 1000)
    hedge = next(r for r in options if r['kind'] == 'hedge')
    assert hedge['shares'] == 100 and hedge['balanced']
    assert hedge['if_held_wins'] == pytest.approx(hedge['if_other_wins'])
    assert hedge['floor'] == pytest.approx(100-60-hedge['added_cost'])
    reduce = next(r for r in options if r['kind'] == 'reduce')
    assert reduce['if_held_wins'] == pytest.approx(reduce['if_other_wins'])
    assert reduce['if_other_wins'] == pytest.approx(reduce['proceeds']-60)
    action = next_action(s, q, 'BOS', 'LAL', 1000, position)
    assert action['floor'] == max(r['floor'] for r in options)
    assert action['floor'] > -60


def test_partial_depth_and_budget_leave_residual_risk(tmp_path):
    s, q = state_and_quotes(tmp_path)
    q['away'].update(asks=[{'price': .4, 'size': 10}], ask=.4, bid=.39, minimum_notional=0)
    h = next(r for r in hedge_options(s, q, {'side': 'home', 'shares': 100, 'cost': 60}, 8) if r['kind'] == 'hedge')
    assert h['shares'] == 10 and not h['balanced']
    assert h['added_cost'] <= 8
    assert h['if_held_wins']-h['if_other_wins'] == 90
    assert h['floor'] < 0  # Improving protection does not imply a profit.


def test_missing_stale_or_different_market_never_recommends_position_trade(tmp_path):
    s, q = state_and_quotes(tmp_path)
    pos = {'side': 'away', 'shares': 100, 'cost': 60}
    q['home']['condition_id'] = 'another-game'
    assert [r['kind'] for r in hedge_options(s, q, pos)] == ['hold']
    assert [r['kind'] for r in hedge_options(s, {}, pos)] == ['hold']
    q['home']['condition_id'] = q['away']['condition_id']
    for row in q.values():
        row['updated_at'] = (NOW-pd.Timedelta(seconds=16)).isoformat()
    assert [r['kind'] for r in hedge_options(s, q, pos)] == ['hold']
    s['quote_state'] = 'final'
    assert next_action(s, q, 'BOS', 'LAL', position=pos)['kind'] == 'finished'


@pytest.mark.parametrize('position', [
    {'side': 'home', 'shares': -1, 'cost': 0}, {'side': 'unknown', 'shares': 1, 'cost': 0},
    {'side': 'away', 'shares': 100, 'cost': 101}, {'side': 'home', 'shares': float('nan'), 'cost': 1},
])
def test_invalid_positions_rejected(tmp_path, position):
    s, q = state_and_quotes(tmp_path)
    with pytest.raises(ValueError):
        hedge_options(s, q, position)


def test_hedge_does_not_require_a_forecast_but_buy_signal_does(tmp_path):
    s, q = state_and_quotes(tmp_path)
    s.update(p_home=None, quote_state='waiting')
    view = match_view({'home_team': 'BOS', 'away_team': 'LAL'}, [], s, q, demo=False)
    assert all(r['market_odds'] for r in view['teams'])
    assert all(r['probability'] is None for r in view['teams'])
    assert view['action']['kind'] == 'wait'
    assert len(hedge_options(s, q, {'side': 'home', 'shares': 100, 'cost': 60})) == 3


def test_sale_validates_fees_and_does_not_invent_depth(tmp_path):
    _, q = state_and_quotes(tmp_path)
    quote = q['home']
    quote['bids'] = []
    assert sell_position(quote, 100)['shares'] == 0
    quote['fee_rate'] = float('nan')
    with pytest.raises(ValueError):
        sell_position(quote, 100)


RULES = ('The final score including any overtime periods determines the winner. '
         'If postponed this market remains open until completed. If canceled entirely it will resolve 50-50.')


def test_auto_discovery_uses_exact_game_and_verifies_standard_rules():
    calls = []
    class Response:
        def __init__(self, data): self.data = data
        def raise_for_status(self): pass
        def json(self): return self.data
    class Session:
        def get(self, url, **kw):
            calls.append((url, kw.get('params')))
            if '/events/slug/' in url:
                return Response({'slug': url.rsplit('/', 1)[1], 'markets': [{**market(), 'description': RULES, 'active': True, 'closed': False, 'acceptingOrders': True}]})
            if '/clob-markets/' in url:
                return Response({})
            token = kw['params']['token_id']
            return Response({'asset_id': token, 'market': 'condition', 'timestamp': int(NOW.timestamp()*1000),
                             'asks': [{'price': '.5', 'size': '100'}], 'bids': [{'price': '.4', 'size': '100'}]})
    game = SimpleNamespace(game_id='g2', home_team='BOS', away_team='LAL', tip_time=NOW)
    reader = PolymarketReader(Session(), lambda: NOW)
    first, second = reader.fetch_game(game), reader.fetch_game(game)
    assert len(first['quotes']) == len(second['quotes']) == 2
    assert first['quotes']['home']['rules_verified']
    assert first['quotes']['away']['token_id'] == 'lakers-token'
    assert sum('/events/slug/' in url for url, _ in calls) == 1
    assert sum('/book' in url for url, _ in calls) == 4  # Cached discovery, fresh books.
    assert calls[0][0].endswith('nba-lal-bos-'+NOW.tz_convert('America/New_York').strftime('%Y-%m-%d'))
    assert not standard_rules({'description': 'Celtics win? Including overtime.'})
    assert not standard_rules({'description': RULES.replace('50-50', 'void')})


def test_discovery_falls_back_to_catalog_but_never_fuzzy_matches():
    game = SimpleNamespace(game_id='g2', home_team='BOS', away_team='LAL', tip_time=NOW)
    class Reader(PolymarketReader):
        def _get(self, url, params=None):
            if '/events/slug/' in url:
                raise requests.HTTPError('404')
            if url.endswith('/events'):
                return [{'slug': 'correct', 'markets': [market()]}, {'slug': 'wrong-day', 'markets': [{**market(), 'gameStartTime': (NOW-pd.Timedelta(days=1)).isoformat()}]}]
            raise AssertionError('Books are replaced by fixture')
        def _fetch_event(self, event, game, slug, rules_verified):
            return {'slug': slug, 'rules': rules_verified}
    assert Reader().fetch_game(game) == {'slug': 'correct', 'rules': None}


def test_scoreboard_maps_espn_aliases_to_nba_ids_and_prefers_live_games():
    class Response:
        def raise_for_status(self): pass
        def json(self):
            def event(i, state, away):
                return {'id': i, 'date': NOW.isoformat(), 'competitions': [{'status': {'type': {'state': state}}, 'competitors': [
                    {'homeAway': 'home', 'team': {'abbreviation': 'NY'}}, {'homeAway': 'away', 'team': {'abbreviation': away}}]}]}
            return {'events': [event('1', 'post', 'GS'), event('2', 'in', 'NO')]}
    class Session:
        def get(self, *args, **kw): return Response()
    games = discover_games(Session())
    assert games[0]['game_id'] == '2'
    assert games[0]['home_team'] == 'NYK' and games[0]['away_team'] == 'NOP'
    assert games[0]['home_team_id'] == 1610612752


def test_live_controller_caches_inputs_not_user_actions_and_waits_on_market_failure(tmp_path, monkeypatch):
    import agents.dashboard_live as module
    s, q = state_and_quotes(tmp_path)
    now = pd.Timestamp.now(tz='UTC')
    s.update(as_of=now.isoformat(), game_id='2')
    for row in q.values():
        row.update(game_id='2', observed_at=now.isoformat(), updated_at=now.isoformat())
    game = dict(game_id='2', tip_time=now.isoformat(), home_team='BOS', away_team='LAL', label='LAL @ BOS')
    monkeypatch.setattr(module, 'discover_games', lambda: [game])
    calls = []
    class Agent:
        latest = {'quote_state': 'live'}
        def poll(self, now):
            calls.append('agent'); return {'snapshot': deepcopy(s)}
    class Market:
        def fetch_game(self, game):
            calls.append('market'); return {'quotes': deepcopy(q), 'errors': []}
    controller = LiveDashboard(market_reader=Market())
    controller.histories['2'] = []
    monkeypatch.setattr(controller, '_agent', lambda game: Agent())
    first = controller.poll('2', 10)
    second = controller.poll('2', 100)
    assert len(calls) == 2 and len(controller.histories['2']) == 1
    assert first['simulation'] is False and second['market_connected']
    assert second['as_of'] >= now.isoformat()
    assert first['teams'][1]['entry_price'] != second['teams'][1]['entry_price']  # Different budget depth.
    with pytest.raises(ValueError):
        controller.poll('wrong-match')
    controller.cache.clear()
    def unavailable(game): raise requests.ConnectionError()
    monkeypatch.setattr(controller.market, 'fetch_game', unavailable)
    failed = controller.poll('2')
    assert failed['action']['kind'] == 'wait' and not failed['market_connected']
    assert all(r['market_odds'] is None for r in failed['teams'])


def test_unverified_live_score_time_blocks_buy_signal_even_with_fitted_model(tmp_path):
    from agents.market_analysis import compare_routes
    s, q = state_and_quotes(tmp_path)
    s['freshness'] = 'source_timestamp_unverified'
    result = compare_routes(s, q, uncertainty_pp=0)
    assert result['candidate'] != 'wait' and result['decision'] == 'wait'
    assert next_action(s, q, 'BOS', 'LAL', demo=False)['kind'] == 'wait'


def test_waiting_tip_explanation_distinguishes_prices_from_missing_forecast(tmp_path):
    s, q = state_and_quotes(tmp_path)
    s.update(p_home=None, quote_state='waiting_for_tip')
    action = next_action(s, q, 'BOS', 'LAL', demo=False)
    assert 'tip-off' in action['reason'] and 'stale' not in action['reason']


def test_dashboard_news_poll_interval_does_not_change_existing_runner_defaults(monkeypatch):
    import agents.dashboard_live as module
    from data_sources.inplay_news import InPlayMedia
    captured = {}
    monkeypatch.setattr(module, 'LiveInPlay', lambda **kw: captured.update(kw) or captured)
    provider = module.dashboard_provider()
    assert provider['media_client'].interval == 5
    assert InPlayMedia(provider['registry']).interval == 60


def test_last_seen_prices_are_labeled_and_never_used_for_actions(tmp_path):
    s, q = state_and_quotes(tmp_path)
    for quote in q.values():
        quote['updated_at'] = (NOW-pd.Timedelta(seconds=40)).isoformat()
    view = match_view({'home_team': 'BOS', 'away_team': 'LAL'}, [{'snapshot': s}], s, q, demo=False)
    assert all(r['market_odds'] and r['price_status'] == 'Last seen' for r in view['teams'])
    assert all(not r['quote_available'] and r['edge_pp'] is None for r in view['teams'])
    assert view['action']['kind'] == 'wait'
    s['market_quotes'] = q
    view = match_view({'home_team': 'BOS', 'away_team': 'LAL'}, [{'snapshot': s}], s, q, demo=False)
    assert view['curve'][0]['market'] is None
