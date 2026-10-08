"""Courtside alert wiring with offline feeds and local HTTP only."""
import json
import re
import threading
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from agents.alerts import AlertConfig, AlertManager
from agents.dashboard_live import LiveDashboard
from agents.dashboard_server import DashboardServer
from test_alerts import snapshot, T0


def test_exported_demo_alerts_use_player_names(tmp_path, monkeypatch):
    from agents import loop_visuals
    from agents.loop_demo import SAMPLE_OUTPUT
    from data_sources import ROOT, read_table

    monkeypatch.setattr(loop_visuals, 'plot_sample', lambda *args: None)
    payload = json.loads((SAMPLE_OUTPUT / 'result.json').read_text(encoding='utf-8'))
    loop_visuals.export_visuals(payload, tmp_path)
    html = (tmp_path / 'demo.html').read_text(encoding='utf-8')
    data = json.loads(re.search(r'<script type="application/json" id="loop-data">(.*?)</script>', html, re.S)[1])
    players = read_table('players', ROOT / 'data' / 'sample')
    names = dict(zip(players.player_id, players.player_name))
    alerts = [a for frame in data['pregame_steps'] for a in frame['view']['alerts']
              if a['type'] == 'injury_status_change']
    assert alerts
    for alert in alerts:
        pid = alert['payload']['player_id']
        assert alert['payload']['player'] == names[pid]
        assert alert['title'].startswith(names[pid] + ' status changed')
        assert alert['message'].startswith(names[pid] + ' is now listed')
        assert f'Player {pid}' not in alert['title']


def test_background_notification_does_not_block_or_lose_read_state(tmp_path):
    started, release = threading.Event(), threading.Event()

    class SlowNotifier:
        def send(self, event):
            started.set()
            assert release.wait(5)
            raise requests.Timeout('secret webhook URL must not leak')

    manager = AlertManager(tmp_path, config=AlertConfig(enabled=True, notify_requested=True,
        webhook_url='https://example.invalid/hook'), mode='live', notifier=SlowNotifier(),
        background_delivery=True)
    try:
        events = manager.process(snapshot(), snapshot(p_home=.54))
        assert started.wait(2)
        assert events[0]['delivery']['status'] == 'pending'
        manager.store.acknowledge([events[0]['event_id']])
    finally:
        release.set()
        manager.close()
    saved = manager.store.latest()[0]
    assert saved['acknowledged']
    assert saved['delivery']['status'] == 'failed'
    assert saved['delivery']['last_error'] == 'Timeout'


def test_live_board_records_same_quotes_once_and_retains_alerts(tmp_path, monkeypatch):
    import agents.dashboard_live as module
    from agents.graph import record_forecaster
    from data_sources.live_news import TableNews
    from test_agent_graph import make_tables, NEWS_AT

    tables = make_tables()
    g = next(tables['games'].iloc[[1]].itertuples(index=False))
    game = {**g._asdict(), 'game_id': '123', 'tip_time': pd.Timestamp.now(tz='UTC') + pd.Timedelta(hours=1)}
    # Real PregameAgent uses a deterministic local model and news provider.
    monkeypatch.setattr(module, 'HistoricalRecordPrior', lambda: record_forecaster)
    monkeypatch.setattr(module, 'LiveNews', lambda **kw: TableNews(tables['news'].iloc[:0]))
    monkeypatch.setattr(module, 'historical_distribution', lambda *a: {
        'total_mean': 220., 'total_sd': 20., 'margin_sd': 14.,
        'history_games': 10, 'history_cutoff': NEWS_AT.isoformat(), 'model': 'test'})

    class Reader:
        calls = 0
        midpoint = .5
        def fetch_bet_contracts(self, game):
            self.calls += 1
            return {'contracts': [{'id': 'condition', 'kind': 'moneyline', 'line': None,
                'quotes': {'home': {'bid': self.midpoint-.01, 'ask': self.midpoint+.01,
                                   'updated_at': pd.Timestamp.now(tz='UTC').isoformat()}}}], 'errors': []}

    reader = Reader()
    controller = LiveDashboard(market_reader=reader, run_root=tmp_path)
    controller.tables = tables
    controller.players = pd.DataFrame({'player_id': [7], 'player_name': ['Test Player']})
    monkeypatch.setattr(controller, 'games', lambda: [game])
    first = controller.betting('123')
    assert not first['alert_notifications_enabled']
    reader.midpoint = .56
    controller.board_cache.clear()
    second = controller.betting('123')
    assert reader.calls == 2
    events = [e for e in second['alerts'] if e['type'] == 'market_move']
    assert len(events) == 1 and events[0]['payload']['market_ticker'] == 'condition:home'
    assert first['own_model']['p_home'] == second['own_model']['p_home']
    agent = controller.pregame_agents['123']
    assert events[0]['event_id'] in agent.latest['alert_ids']
    controller.acknowledge_alerts('123', [events[0]['event_id']])
    cached = controller.betting('123')
    assert reader.calls == 2 and cached['alert_unread_count'] == 0
    agent.alert_manager.close()
    restored = LiveDashboard(market_reader=reader, run_root=tmp_path)
    assert restored.alert_history('123')[0]['acknowledged']
    assert restored.alert_history('456') == []


def test_quotes_unavailable_recovery_and_unknown_age(tmp_path):
    manager = AlertManager(tmp_path)
    unavailable = LiveDashboard._alert_market_snapshot({'contracts': []}, T0)
    old = snapshot(polymarket=unavailable)
    assert manager.process(None, old)[0]['type'] == 'quote_stale'
    assert manager.process(old, snapshot(at=T0 + pd.Timedelta(seconds=5), polymarket=unavailable)) == []
    fresh = LiveDashboard._alert_market_snapshot({'contracts': [{'id': 'm', 'quotes': {
        'home': {'bid': .4, 'ask': .5, 'updated_at': T0.isoformat()}}}]}, T0)
    events = manager.process(old, snapshot(at=T0 + pd.Timedelta(seconds=10), polymarket=fresh))
    assert [e['type'] for e in events] == ['quote_recovered']


def test_http_alerts_ack_download_and_current_template(tmp_path):
    controller = LiveDashboard(run_root=tmp_path)
    manager = AlertManager(tmp_path / '123' / 'pregame')
    manager.process(snapshot(), snapshot(p_home=.54))
    controller.pregame_agents['123'] = SimpleNamespace(alert_manager=manager)
    server = DashboardServer(('127.0.0.1', 0), controller)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        html = requests.get(base, timeout=5).text
        assert 'id="alertcard"' in html and 'Trigger conditions &amp;' not in html
        rows = requests.get(base + '/api/alerts?game_id=123', timeout=5).json()['alerts']
        body = {'game_id': '123', 'event_ids': [rows[0]['event_id']]}
        assert requests.post(base + '/api/alerts/acknowledge', json=body,
                             headers={'Origin': 'https://unrelated.example'}, timeout=5).status_code == 403
        result = requests.post(base + '/api/alerts/acknowledge', json=body, timeout=5)
        assert result.json() == {'acknowledged': 1}
        download = requests.get(base + '/api/alerts?game_id=123&download=1', timeout=5)
        assert json.loads(download.text)['acknowledged']
        assert requests.get(base + '/api/alerts?game_id=../', timeout=5).status_code == 400
        assert requests.post(base + '/api/alerts/acknowledge', json=[], timeout=5).status_code == 400
    finally:
        server.shutdown()
        worker.join(5)
        server.server_close()
        controller.market.close()


def test_dashboard_requires_explicit_notify_even_with_enabled_config(tmp_path):
    config = AlertConfig(enabled=True, notify_requested=True, webhook_url='https://example.invalid/hook')
    dashboard = LiveDashboard(alert_config=config, run_root=tmp_path)
    assert not dashboard.alert_config.can_notify('live')
    enabled = LiveDashboard(alert_config=config, notify=True, run_root=tmp_path)
    assert enabled.alert_config.can_notify('live')
    assert not enabled.alert_config.can_notify('replay')
    dashboard.market.close()
    enabled.market.close()
