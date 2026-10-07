"""English match dashboard with automatic live APIs: python -m agents.dashboard_server."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agents.dashboard_live import LiveDashboard
from agents.loop_demo import SAMPLE_OUTPUT


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
                return self._send(200, (SAMPLE_OUTPUT/'demo.html').read_bytes(), 'text/html')
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--model-file', type=Path, help='Trusted locally trained in-play model')
    args = parser.parse_args()
    server = DashboardServer(('127.0.0.1', args.port), LiveDashboard(model_file=args.model_file))
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
