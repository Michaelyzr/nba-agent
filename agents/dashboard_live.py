"""Automatic ESPN match discovery, independent news agent and Polymarket polling."""
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import requests
from nba_api.stats.static import teams

from agents.inplay import InPlayAgent
from agents.alerts import AlertConfig, AlertManager, AlertStore
from agents.pregame import PregameAgent
from agents.betting import make_board
from agents.graph import clip, log5_home, win_rate, usual_minutes, OUT_MINUTE_VALUE
from agents.match_view import match_view, hedge_options
from agents.market_analysis import _finite
from data_sources import ROOT, FROZEN, read_table
from data_sources.inplay import LiveInPlay
from data_sources.live_news import LiveNews
from data_sources.inplay_news import InPlayMedia
from data_sources.inplay_score import ESPN_BASE, team_code
from data_sources.inplay_types import utc
from data_sources.news_registry import NewsRegistry
from data_sources.polymarket_books import PolymarketReader
from forecast.betting import historical_distribution, distribution


class HistoricalRecordPrior:
    name = 'historical-record-prototype'
    def win(self, view, game, availability=None):
        done = view.games().dropna(subset=['home_pts'])
        p = log5_home(win_rate(done, game.home_team_id), win_rate(done, game.away_team_id))
        for pid, share in (availability or {}).items():
            team, minutes = usual_minutes(view.player_games(), int(pid))
            p += (-1 if team == game.home_team_id else 1 if team == game.away_team_id else 0)*OUT_MINUTE_VALUE*minutes*share
        return {'p_home': clip(p)}


def dashboard_provider():
    registry = NewsRegistry.load()
    return LiveInPlay(registry=registry, media_client=InPlayMedia(registry, interval_seconds=5))


def discover_games(session=None):
    session = session or requests.Session()
    now = pd.Timestamp.now(tz='America/New_York')
    # ESPN's undated endpoint can keep the previous day's completed slate.
    dates = sorted({(now+pd.Timedelta(hours=h)).strftime('%Y%m%d') for h in (-8,0,24)})
    def fetch(date):
        response = session.get(ESPN_BASE+'/scoreboard', params={'dates':date}, timeout=10)
        response.raise_for_status()
        return response.json().get('events', [])
    with ThreadPoolExecutor(max_workers=3) as pool:
        events = {str(event['id']):event for group in pool.map(fetch,dates) for event in group}
    known = {r['abbreviation']: r for r in teams.get_teams()}
    result = []
    for event in events.values():
        competition = event.get('competitions', [{}])[0]
        sides = {r.get('homeAway'): r for r in competition.get('competitors', [])}
        if not {'home', 'away'} <= set(sides):
            continue
        home, away = [team_code(sides[k]['team']['abbreviation']) for k in ('home', 'away')]
        if home not in known or away not in known:
            continue
        tip = utc(event['date'])
        result.append({'game_id': str(event['id']), 'tip_time': tip.isoformat(),
                       'date': tip.tz_convert('America/New_York').strftime('%Y-%m-%d'),
                       'home_team': home, 'away_team': away, 'home_team_id': known[home]['id'], 'away_team_id': known[away]['id'],
                       'phase': competition.get('status', event.get('status', {})).get('type', {}).get('state', 'pre'),
                       'label': away+' @ '+home})
    return sorted(result, key=lambda g: ({'in': 0, 'pre': 1, 'post': 2}.get(g['phase'], 3), g['tip_time']))


class LiveDashboard:
    """One serialized refresh per match, cached for five seconds across viewers."""
    def __init__(self, source=None, model_file=None, market_reader=None, provider_factory=None,
                 notify=False, alert_config=None, run_root=None):
        self.source = source or (FROZEN if all((FROZEN/(n+'.parquet')).exists() for n in ('games', 'players', 'player_games')) else ROOT/'data'/'sample')
        self.tables = None
        self.players = None
        self.model_file = model_file
        self.market = market_reader or PolymarketReader()
        self.provider_factory = provider_factory or dashboard_provider
        self.agents, self.histories, self.cache = {}, {}, {}
        self.pregame_agents, self.board_cache, self.distributions = {}, {}, {}
        self.catalog, self.catalog_at = [], float("-inf")
        self.lock = threading.RLock()
        self.board_locks = {}
        self.run_root = Path(run_root) if run_root else ROOT/'runs'/'dashboard'/('live-'+uuid4().hex[:8])
        self.alert_config = (alert_config or AlertConfig.from_env()).for_mode('live', notify)

    def games(self):
        with self.lock:
            if time.monotonic()-self.catalog_at > 30:
                now = pd.Timestamp.now(tz='UTC')
                self.catalog = [g for g in discover_games() if now-utc(g['tip_time']) <= pd.Timedelta(hours=8)]
                self.catalog_at = time.monotonic()
            return self.catalog

    def _load_tables(self):
        if self.tables is None:
            self.tables = {n: read_table(n, self.source) for n in ('games', 'player_games')}
            self.tables['news'] = pd.DataFrame()
            self.players = read_table('players', self.source)

    def _agent(self, game):
        self._load_tables()
        if game.game_id not in self.agents:
            model = pickle.loads(Path(self.model_file).read_bytes()) if self.model_file else None
            self.agents[game.game_id] = InPlayAgent(self.tables, self.players, game, self.provider_factory(),
                                                   self.run_root/game.game_id, prior_model=HistoricalRecordPrior(),
                                                   win_model=model, clock=lambda: pd.Timestamp.now(tz='UTC'))
            self.histories.setdefault(game.game_id, [])
        return self.agents[game.game_id]

    @staticmethod
    def _alert_market_snapshot(market, observed_at):
        """Convert the betting board's quotes to the alert detector schema."""
        contracts = (market or {}).get('contracts', [])
        rows = []
        ages = []
        for contract in contracts:
            for side, quote in (contract.get('quotes') or {}).items():
                quote = dict(quote or {})
                bid, ask = quote.get('bid'), quote.get('ask')
                midpoint = quote.get('midpoint')
                if midpoint is None and bid is not None and ask is not None:
                    midpoint = (float(bid) + float(ask)) / 2
                updated = quote.get('updated_at') or quote.get('observed_at')
                age = None
                if updated:
                    try:
                        age = max(0.0, (utc(observed_at) - utc(updated)).total_seconds())
                        ages.append(age)
                    except (TypeError, ValueError):
                        pass
                rows.append({
                    'market_ticker': f"{contract.get('id') or contract.get('condition_id') or 'market'}:{side}",
                    'team': quote.get('team') or side,
                    'bid': bid, 'ask': ask, 'midpoint': midpoint,
                    'updated_at': updated, 'age_seconds': age,
                })
        errors = list((market or {}).get('errors') or [])
        complete = rows and len(ages) == len(rows) and all(r['midpoint'] is not None for r in rows)
        status = 'OK' if complete and not errors else ('WARNING' if rows else 'UNAVAILABLE')
        return {'source': 'polymarket', 'status': status,
                'fetched_at': utc(observed_at).isoformat(),
                'age_seconds': max(ages) if ages else None,
                'markets': rows, 'errors': errors}

    def _pregame_agent(self, game):
        if game.game_id not in self.pregame_agents:
            output = self.run_root / game.game_id / 'pregame'
            manager = AlertManager(output, config=self.alert_config, mode='live', run_id=self.run_root.name,
                                   players=self.players, background_delivery=True)
            self.pregame_agents[game.game_id] = PregameAgent(
                self.tables, self.players, game, HistoricalRecordPrior(),
                LiveNews(timeout=3, report_slots=1), output,
                clock=lambda: pd.Timestamp.now(tz='UTC'), alert_manager=manager,
                alert_mode='live')
        return self.pregame_agents[game.game_id]

    def poll(self, game_id, budget=100, position=None):
        budget = _finite(budget, 'budget', lo=1, hi=100000)
        if position:
            hedge_options({}, {}, position, budget)
        with self.lock:
            game_lock = self.board_locks.setdefault(str(game_id), threading.RLock())
            self._load_tables()
        with game_lock:
            games = self.games()
            found = next((g for g in games if g['game_id'] == str(game_id)), None)
            if not found:
                raise ValueError('This match is not on the current NBA schedule')
            game = SimpleNamespace(**{**found, 'tip_time':utc(found['tip_time'])})
            if utc(pd.Timestamp.now(tz='UTC'))-utc(game.tip_time) > pd.Timedelta(hours=8):
                raise ValueError('This game is archived; choose a current match')
            cached = self.cache.get(game.game_id)
            if not cached or time.monotonic()-cached[0] >= 5:
                agent = self._agent(game)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    model_job = pool.submit(agent.poll, pd.Timestamp.now(tz='UTC'))
                    market_job = pool.submit(self.market.fetch_game, game)
                    snapshot = dict(model_job.result()['snapshot'])
                    try:
                        market = market_job.result()
                    except (requests.RequestException, ValueError, TypeError, KeyError):
                        market = {'quotes': {}, 'errors': ['Polymarket is not available for this game yet.']}
                analysis_at = pd.Timestamp.now(tz='UTC')
                snapshot['model_as_of'] = snapshot['as_of']
                model_age = (analysis_at-utc(snapshot['as_of'])).total_seconds()
                snapshot['as_of'] = analysis_at.isoformat()
                if model_age > 15 and snapshot.get('quote_state') == 'live':
                    snapshot.update(p_home=None, p_away=None, quote_state='stale_score')
                snapshot['market_quotes'] = market.get('quotes', {})
                if agent.latest.get('quote_state') == 'final':
                    snapshot['market_quotes'] = {}
                history = self.histories[game.game_id]
                if not history or history[-1]['snapshot']['model_as_of'] != snapshot['model_as_of']:
                    history.append({'phase': 'inplay', 'snapshot': snapshot})
                    if len(history) > 5000:
                        history.pop(0)
                self.cache[game.game_id] = (time.monotonic(), snapshot, market)
            _, snapshot, market = self.cache[game.game_id]
            payload = {'home_team': game.home_team, 'away_team': game.away_team}
            view = match_view(payload, self.histories[game.game_id], snapshot, snapshot['market_quotes'], False, budget, position)
            view['connection_note'] = '; '.join(market.get('errors', []))
            if not view['connection_note'] and view['market_connected'] and not any(r['quote_available'] for r in view['teams']):
                view['connection_note'] = 'Polymarket connected. Last-seen prices are for reference; waiting for a fresh book before any action.'
            view['history_source'] = 'Local historical NBA statistics'
            view['auto_refresh_seconds'] = 5
            return view

    def betting(self, game_id):
        """Own-model betting board, with isolated pregame and in-play agent state."""
        with self.lock:
            game_lock = self.board_locks.setdefault(str(game_id), threading.RLock())
            self._load_tables()
        with game_lock:
            found = next((g for g in self.games() if g['game_id'] == str(game_id)), None)
            if not found:
                raise ValueError('Select a current NBA game')
            game = SimpleNamespace(**{**found, 'tip_time':utc(found['tip_time'])})
            self._load_tables()
            now = pd.Timestamp.now(tz='UTC')
            phase = 'pregame' if now < utc(game.tip_time) else 'inplay'
            key = (game.game_id, phase)
            cached = self.board_cache.get(key)
            if not cached or time.monotonic()-cached[0] >= 5:
                if phase == 'pregame':
                    agent = self._pregame_agent(game)
                else:
                    agent = self._agent(game)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    reference = pool.submit(self.market.fetch_bet_contracts, game)
                    if phase == 'pregame':
                        # Reuse this poll's order books. Forecast features never consume
                        # them; PregameAgent attaches them after calculating probability.
                        def alert_quotes(_game, observed):
                            try:
                                fresh_market = reference.result()
                            except (requests.RequestException, ValueError, TypeError, KeyError):
                                fresh_market = {'contracts': [], 'errors': ['Polymarket reference unavailable.']}
                            return self._alert_market_snapshot(fresh_market, observed)
                        agent.market_provider = alert_quotes
                    prediction = pool.submit(agent.poll, now)
                    result = prediction.result()
                    raw = result.get('snapshot') or agent.latest
                    if raw is None:
                        raise ValueError('Tip-off transition; waiting for current game data')
                    snapshot = dict(raw)
                    if phase == 'pregame':
                        snapshot.update(quote_state='pregame', model_trained=False, quote_quality='provisional')
                    try:
                        market = reference.result()
                    except (requests.RequestException, ValueError, TypeError, KeyError):
                        market = {'contracts': [], 'errors': ['Polymarket reference unavailable. Retrying automatically.']}
                at = pd.Timestamp.now(tz='UTC')
                snapshot['model_as_of'] = snapshot['as_of']
                age = (at-utc(snapshot['as_of'])).total_seconds()
                snapshot['as_of'] = at.isoformat()
                if phase == 'pregame' and at >= utc(game.tip_time):
                    snapshot.update(p_home=None, quote_state='waiting_for_tip')
                if age > 15 and snapshot.get('quote_state') in ('live', 'pregame'):
                    snapshot.update(p_home=None, quote_state='stale_score')
                if game.game_id not in self.distributions:
                    self.distributions[game.game_id] = historical_distribution(self.tables['games'], game, now)
                own = distribution(self.distributions[game.game_id], snapshot)
                board = make_board(game, snapshot, own, market.get('contracts', []), False)
                self.board_cache[key] = (time.monotonic(), snapshot, board, market)
                history = self.histories.setdefault(game.game_id, [])
                if not history or history[-1]['snapshot']['as_of'] != snapshot['as_of']:
                    history.append({'phase': phase, 'snapshot': snapshot})
                    del history[:-5000]
            _, snapshot, board, market = self.board_cache[key]
            view = match_view({'home_team': game.home_team, 'away_team': game.away_team},
                              self.histories[game.game_id], snapshot, {}, False)
            past = [r['snapshot'] for r in self.histories[game.game_id] if r['phase'] == 'pregame' and r['snapshot'].get('p_home') is not None]
            view['pre_curve'] = [{'minute': (utc(s['as_of'])-utc(past[0]['as_of'])).total_seconds()/60, 'p': s['p_home']} for s in past] if past else []
            view.update(board, phase=phase, market_connected=any(r['reference_quote'] for r in board['bets']),
                        connection_note='; '.join(market.get('errors', [])), auto_refresh_seconds=5, tip_time=utc(game.tip_time).isoformat())
            view['alerts'] = self.alert_history(game.game_id)
            view['alert_unread_count'] = sum(not row.get('acknowledged', False) for row in view['alerts'])
            view['alert_notifications_enabled'] = self.alert_config.can_notify('live')
            view['alert_error'] = snapshot.get('alert_error')
            return view

    def alert_history(self, game_id):
        if not str(game_id).isdigit():
            raise ValueError('Select a current NBA game')
        agent = self.pregame_agents.get(str(game_id))
        manager = getattr(agent, 'alert_manager', None)
        if manager:
            return manager.store.latest()
        return AlertStore(self.run_root / str(game_id) / 'pregame').latest()

    def acknowledge_alerts(self, game_id, event_ids):
        if not str(game_id).isdigit():
            raise ValueError('Select a current NBA game')
        with self.lock:
            game_lock = self.board_locks.setdefault(str(game_id), threading.RLock())
        with game_lock:
            agent = self.pregame_agents.get(str(game_id))
            manager = getattr(agent, 'alert_manager', None)
            store = manager.store if manager else AlertStore(self.run_root / str(game_id) / 'pregame')
            return {'acknowledged': store.acknowledge(event_ids)}

    def close(self):
        for agent in self.agents.values():
            agent.provider.close()
        for agent in self.pregame_agents.values():
            agent.provider.session.close()
            agent.alert_manager.close()
        self.market.close()
