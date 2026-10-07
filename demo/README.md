# Agent thought-process demo

`demo_app.py` replays precomputed traces of every agent (trader LangGraph, review loop, LLM tool agent +
priced-in sceptic, Coach, orchestrator, pregame, in-play). It reads only `demo/traces/*.json`, so it needs
no data, models or API keys.

## Run locally

```bash
pip install -r requirements-demo.txt
streamlit run demo_app.py --server.port 8601
```

## Quick public URL (laptop must stay on)

```bash
streamlit run demo_app.py --server.headless true --server.port 8601 &
cloudflared tunnel --no-autoupdate --url http://localhost:8601    # prints https://<random>.trycloudflare.com
```

No account is needed. If Homebrew is missing, download the binary:
`curl -sSL -o cf.tgz https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64.tgz && tar xzf cf.tgz`.
The URL changes every time the tunnel restarts, and it stops when the laptop sleeps or the process ends.
Fallbacks: `npx localtunnel --port 8601` or `ssh -R 80:localhost:8601 nokey@localhost.run`.

## Durable hosting: Streamlit Community Cloud (free, about 3 minutes)

The repo `Michaelyzr/nba-agent` is public, so no extra GitHub authorisation is needed beyond signing in.

1. Go to https://share.streamlit.io and click **Continue with GitHub**; sign in as the repo owner (or a collaborator).
2. Click **Create app** (top right), then **Deploy a public app from GitHub** (or "Yup, I have an app").
3. Fill in:
   - Repository: `Michaelyzr/nba-agent`
   - Branch: `models/m1-baselines`
   - Main file path: **`demo/streamlit_app.py`** (a two-line wrapper that runs `demo_app.py`). Use this rather
     than `demo_app.py`: Community Cloud installs the `requirements.txt` next to the main file first, and
     `demo/requirements.txt` is just streamlit + pandas, whereas the root `requirements.txt` pulls in torch and
     may exceed the free tier.
   - App URL: pick a subdomain, e.g. `nba-agent-demo`.
4. Click **Advanced settings**: set Python version **3.12**, and leave Secrets **empty** (none are needed).
5. Click **Deploy**. The first build takes 1-3 minutes; the app is then at `https://<subdomain>.streamlit.app`.
   It sleeps after a few days without visitors; open it once before the presentation to wake it.

## Rebuild the traces (needs the full local setup: data/frozen, models/, the venv)

```bash
PYTHONPATH=. python demo/build_traces.py       # about 25 s; writes demo/traces/*.json (~130 KB)
pytest -q tests/test_demo_traces.py            # loads, required steps, no secrets, no look-ahead, AppTest smoke
```

Scenarios: CLE @ POR, 1 Feb 2026 (Avdija out, POR "no" at 55c, CLV +7.5c); the same decision with a planted
future-news citation (checks reject, retry passes), with planted "lock" wording (blocked), and with the kill
switch tripped; MIL @ BOS (pass); POR @ UTA (learned rule r023 vetoes a trade); IND @ BKN (stub-LLM sceptic
rejects); three real Gemini tool-agent decisions replayed from the response cache (all passes; the
other cached Gemini decisions hit API quota errors). Tool-agent buys, validation and the sceptic are shown with the
deterministic stub LLM, labelled as such. In-play uses the synthetic game script from `data_sources/inplay_demo.py`.
