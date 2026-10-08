"""English match dashboard with automatic live APIs: python -m agents.dashboard_server."""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agents.dashboard_live import LiveDashboard
from agents.loop_demo import SAMPLE_OUTPUT
from agents.alerts import AlertConfig


def dashboard_html():
    """Use current templates with the bundled replay, without regenerating plots."""
    bundled = (SAMPLE_OUTPUT / 'demo.html').read_text(encoding='utf-8')
    data = re.search(r'<script type="application/json" id="loop-data">(.*?)</script>', bundled, re.S)
    if data is None:
        raise ValueError('Bundled replay data is missing')
    templates = Path(__file__).parent / 'templates'
    return (templates.joinpath('loop_demo.html').read_text(encoding='utf-8')
            .replace('__LOOP_DATA__', data[1])
            .replace('__ANALYSIS_SCRIPT__', templates.joinpath('betting_math.js').read_text(encoding='utf-8'))
            .replace('__DASHBOARD_SCRIPT__', templates.joinpath('match_dashboard.js').read_text(encoding='utf-8'))
            .encode('utf-8'))


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, controller=None):
        self.controller = controller or LiveDashboard()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', content_type+'; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        origin = self.headers.get('Origin', '')
        if origin == 'null' or origin.startswith(('http://localhost:', 'http://127.0.0.1:')):
            self.send_header('Access-Control-Allow-Origin', origin)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urlparse(self.path)
        try:
            if route.path == '/api/health':
                result = {'status': 'ok', 'refresh_seconds': 5}
            elif route.path == '/api/games':
                result = {'games': self.server.controller.games()}
            elif route.path == '/api/betting':
                params = parse_qs(route.query)
                game_id = params.get('game_id', [''])[0]
                if not game_id.isdigit():
                    raise ValueError('Select a current NBA game')
                result = self.server.controller.betting(game_id)
            elif route.path == '/api/match':
                params = parse_qs(route.query)
                game_id = params.get('game_id', [''])[0]
                if not game_id.isdigit():
                    raise ValueError('Select a current NBA game')
                budget = float(params.get('budget', ['100'])[0])
                position = None
                if params.get('position_side'):
                    position = {'side': params['position_side'][0], 'shares': float(params.get('position_shares', ['0'])[0]),
                                'cost': float(params.get('position_cost', ['0'])[0])}
                result = self.server.controller.poll(game_id, budget, position)
            elif route.path in ('/', '/demo.html'):
                return self._send(200, dashboard_html(), 'text/html')
            elif route.path == '/api/alerts':
                game_id = parse_qs(route.query).get('game_id', [''])[0]
                if not game_id.isdigit():
                    raise ValueError('Select a current NBA game')
                rows = self.server.controller.alert_history(game_id)
                if parse_qs(route.query).get('download') == ['1']:
                    body = ''.join(json.dumps(row, ensure_ascii=False, default=str) + '\n' for row in rows)
                    return self._send(200, body.encode('utf-8'), 'application/x-ndjson')
                result = {'alerts': rows}
            elif route.path == '/sample.png':
                return self._send(200, (SAMPLE_OUTPUT/'sample.png').read_bytes(), 'image/png')
            else:
                return self._send(404, b'{"error":"Not found"}')
            self._send(200, json.dumps(result, ensure_ascii=False, default=str, allow_nan=False).encode())
        except ValueError as exc:
            self._send(400, json.dumps({'error': str(exc)}).encode())
        except Exception:
            self._send(503, b'{"error":"Live data is temporarily unavailable. Please try again."}')

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path != '/api/alerts/acknowledge':
            return self._send(404, b'{"error":"Not found"}')
        # A local browser may mark alerts read; arbitrary websites may not mutate state.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self._send(403, b'{"error":"Use the dashboard service URL"}')
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 65536 or not self.headers.get('Content-Type', '').startswith('application/json'):
                raise ValueError('Expected a bounded JSON request')
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError('Expected an object')
            game_id, ids = body.get('game_id'), body.get('event_ids')
            if not isinstance(game_id, str) or not game_id.isdigit():
                raise ValueError('Select a current NBA game')
            if not isinstance(ids, list) or len(ids) > 500 or not all(isinstance(i, str) for i in ids):
                raise ValueError('Expected up to 500 event IDs')
            result = self.server.controller.acknowledge_alerts(game_id, ids)
            self._send(200, json.dumps(result).encode())
        except (ValueError, TypeError):
            self._send(400, b'{"error":"Invalid acknowledgement request"}')
        except Exception:
            self._send(503, b'{"error":"Could not save read state; please retry"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--model-file', type=Path, help='Trusted locally trained in-play model')
    parser.add_argument('--notify', action='store_true', help='Enable live Webhooks when ALERT_ENABLED and a URL are configured')
    parser.add_argument('--alert-config', type=Path, help='Optional alert configuration JSON')
    parser.add_argument('--run-root', type=Path, help='Reuse a run directory to retain alert history across restarts')
    args = parser.parse_args()
    server = DashboardServer(('127.0.0.1', args.port), LiveDashboard(model_file=args.model_file,
        notify=args.notify, alert_config=AlertConfig.from_env(args.alert_config), run_root=args.run_root))
    print(f'Match dashboard: http://127.0.0.1:{args.port} | live polling every 5 seconds', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.controller.close()
        server.server_close()


if __name__ == '__main__':
    main()
