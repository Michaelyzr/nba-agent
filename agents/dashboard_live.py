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
from agents.graph import clip, log5_home, win_rate
from agents.match_view import match_view, hedge_options
from agents.market_analysis import _finite
from data_sources import ROOT, FROZEN, read_table
from data_sources.inplay import LiveInPlay
from data_sources.inplay_news import InPlayMedia
from data_sources.inplay_score import ESPN_BASE, team_code
from data_sources.inplay_types import utc
from data_sources.news_registry import NewsRegistry
from data_sources.polymarket_books import PolymarketReader


class HistoricalRecordPrior:
    name = 'historical-record-prototype'
    def win(self, view, game):
        done = view.games().dropna(subset=['home_pts'])
        return {'p_home': clip(log5_home(win_rate(done, game.home_team_id), win_rate(done, game.away_team_id)))}


def dashboard_provider():
    registry = NewsRegistry.load()
    return LiveInPlay(registry=registry, media_client=InPlayMedia(registry, interval_seconds=5))


def discover_games(session=None):
    session = session or requests.Session()
    response = session.get(ESPN_BASE+'/scoreboard', timeout=10)
    response.raise_for_status()
    known = {r['abbreviation']: r for r in teams.get_teams()}
    result = []
    for event in response.json().get('events', []):
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
    def __init__(self, source=None, model_file=None, market_reader=None, provider_factory=None):
        self.source = source or (FROZEN if all((FROZEN/(n+'.parquet')).exists() for n in ('games', 'players', 'player_games')) else ROOT/'data'/'sample')
        self.tables = None
        self.players = None
        self.model_file = model_file
        self.market = market_reader or PolymarketReader()
        self.provider_factory = provider_factory or dashboard_provider
        self.agents, self.histories, self.cache = {}, {}, {}
        self.catalog, self.catalog_at = [], float("-inf")
        self.lock = threading.RLock()
        self.run_root = ROOT/'runs'/'dashboard'/('live-'+uuid4().hex[:8])

    def games(self):
        with self.lock:
            if time.monotonic()-self.catalog_at > 30:
                now = pd.Timestamp.now(tz='UTC')
                self.catalog = [g for g in discover_games() if now-utc(g['tip_time']) <= pd.Timedelta(hours=8)]
                self.catalog_at = time.monotonic()
            return self.catalog

    def _agent(self, game):
        if self.tables is None:
            self.tables = {n: read_table(n, self.source) for n in ('games', 'player_games')}
            self.tables['news'] = pd.DataFrame()
            self.players = read_table('players', self.source)
        if game.game_id not in self.agents:
            model = pickle.loads(Path(self.model_file).read_bytes()) if self.model_file else None
            self.agents[game.game_id] = InPlayAgent(self.tables, self.players, game, self.provider_factory(),
                                                   self.run_root/game.game_id, prior_model=HistoricalRecordPrior(),
                                                   win_model=model, clock=lambda: pd.Timestamp.now(tz='UTC'))
            self.histories[game.game_id] = []
        return self.agents[game.game_id]

    def poll(self, game_id, budget=100, position=None):
        budget = _finite(budget, 'budget', lo=1, hi=100000)
        if position:
            hedge_options({}, {}, position, budget)
        with self.lock:
            games = self.games()
            found = next((g for g in games if g['game_id'] == str(game_id)), None)
            if not found:
                raise ValueError('This match is not on the current NBA schedule')
            game = SimpleNamespace(**found)
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

    def close(self):
        for agent in self.agents.values():
            agent.provider.close()
        self.market.close()
