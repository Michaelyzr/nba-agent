# Market-Graded Learning for NBA Game-Impact Intelligence

DASC7606C Track 2 project: a self-improving multi-agent system that turns
time-sensitive NBA news and statistics into cited impact briefs, calibrated
player and game forecasts, and market-graded evaluations.

The target product answers **what changed, who is affected, and how sure are
we?** after events such as an injury or rest update, minutes restriction,
lineup or role change, trade, or back-to-back. It serves fantasy and
sports-media platforms, broadcasters, team analysts, and retail users through
channel-specific outputs and risk policies.

> **Project status:** this repository currently contains the earlier
> single-agent cited-stat prototype. The three-agent market-graded system,
> historical replay, GRU forecast, rule gate, and live/paper trading workflow
> described below are the approved target architecture and are not yet
> implemented. Do not present planned components as completed results.

## Target workflow

```text
Late news + frozen NBA statistics
              |
              v
      Forecaster agent
  cited minutes, points and win distributions
              |
              v
  Market grader / trader agent
   model probability vs recorded price
              |
              v
  Deterministic risk limits and paper/live fill
              |
              v
         Game settles
              |
              v
        Reviewer agent
      error cause + proposed rule
              |
              v
  Deterministic historical back-test gate
       | accepted             | rejected
       v                      v
  Rule notebook          discard with reason
       |
       +--------> later forecasts and decisions
```

The agent that proposes a rule cannot approve it. Calculations, risk limits,
leakage checks, and the approval gate are deterministic code rather than LLM
judgments.

## Planned components

| Component | Type | Responsibility |
| --- | --- | --- |
| Forecaster | LLM agent | Interpret time-stamped news, select NBA statistics, call forecast models, and produce cited channel-specific briefs |
| Market grader / trader | LLM agent | Compare calibrated probabilities with Kalshi or Polymarket prices and propose a paper or live order with a cited reason |
| Reviewer | LLM agent | After settlement, classify mistakes and propose a machine-checkable rule |
| Risk limits | Code | Enforce stake, game and daily caps; liquidity limits; no post-tip orders or parlays; drawdown stop; retail confirmation |
| Rule gate | Code | Back-test proposed rules using information available before each decision and retain only rules that improve a frozen metric |
| Replay and scorer | Code | Replay the season chronologically and report calibration, closing-line value, P&L, drawdown and policy failures |

## Deep-learning plan

- **Play classifier:** estimate whether a player will play.
- **Minutes and points model:** a GRU over the player's previous 20 games plus
  rest, unavailable teammates, opponent pace, and other pre-game context. It
  outputs a distribution rather than one point estimate.
- **Win model:** update team strength for player availability.
- **Baselines:** a 10-game rolling average and gradient boosting with
  teammates-out features.

The GRU distribution supports questions such as the probability that a player
scores over 22.5 points. The LLM does not calculate those probabilities.

## Data plan

| Data | Source |
| --- | --- |
| Game logs, advanced and tracking box scores, play-by-play, lineups and on/off statistics | `nba_api`, frozen locally and filtered to the decision timestamp |
| Time-stamped availability, lineup and role information | NBA injury-report PDFs, team reports and selected beat notes; inactive lists as fallback |
| Game-winner and player-prop price history | Kalshi historical API; Polymarket price history as backup |
| Missing historical price depth | A documented third-party archive only if required |

API keys belong in an untracked `.env` file. Generated data and trained model
artifacts are also excluded from git.

## Evaluation

Development data runs from October through January. February through April is
the frozen test period; player props use only the months for which reliable
historical prices exist.

Compare:

1. the full agent with learning;
2. the same agent with learning disabled;
3. the forecasting model with a fixed model-price threshold and no agent;
4. a fixed minutes-allocation rule;
5. a chat model given the same evidence on a documented sample.

Primary metrics are forecast calibration (Brier score) and closing-line value.
Secondary metrics are profit after fees, maximum drawdown, reviewer error-cause
accuracy, and policy-test pass rate. Flat or negative market performance must
be reported rather than reframed as an edge.

## Safety

Live execution is optional and must pass non-overridable code checks:

- caps per order, game and day;
- displayed-liquidity and fee-aware fills;
- no post-tip orders or parlays;
- maximum-drawdown kill switch and loss cool-down;
- no order when the model-market gap or recent calibration misses its gate;
- explicit retail confirmation for every live order;
- no wagering functionality in team or media channels;
- venue age, location, account and legal requirements.

The interface must show distributions, timestamps and citations. It must not
use “lock,” “guaranteed,” or similar false-certainty language.

## Current runnable prototype

The current code implements one LangGraph agent for cited questions over saved
NBA-shaped data. Every number cites saved games, every quotation cites a saved
paragraph, and a signing estimate is labelled as a model prediction with its
held-out error. Its data is synthetic: real team names with made-up players,
statistics, contracts and news.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python make_sample_data.py
python train_signing.py

# Optional cross-encoder fine-tunes; each downloads a base model
python train_ranker.py
python train_support.py

cp .env.example .env
```

Without `GEMINI_API_KEY`, the chat steps use offline rules and templates.

### Run

```bash
python graph.py "Did <player>'s true shooting change after he was traded this season?"
python graph.py --plant playoffs "<same question>"
streamlit run app.py
python evaluate.py
pytest -q
python export_csv.py
```

Player names and supported questions are generated in
`data/eval/questions.json`.

### Existing files

| File | Current role |
| --- | --- |
| `graph.py` | Single-agent LangGraph state, routing, retries and clarification |
| `steps.py` | Clarify, plan, rank, generate/execute code, predict, check and write nodes |
| `checks.py` | Deterministic checks for season, team, rate basis, game count, statistics, citations and quotations |
| `models.py` | BM25/cross-encoder retrieval, support checking and signing network |
| `llm.py` | Gemini wrapper with offline fallbacks |
| `skills/*.md` | Question-specific table and calculation instructions |
| `download_season.py` | Resumable `nba_api` season download; not yet validated end to end |
| `make_sample_data.py` | Synthetic data, labels and evaluation questions |
| `train_*.py`, `evaluate.py` | Prototype models, baselines and metrics |

Faults accepted by `--plant` on the first attempt are `playoffs`,
`wrong_team`, `no_count`, `rate_mix`, `invented_quote`, `wrong_paragraph`,
`no_error`, and `note_number`.

## Current limitations

- The repository does not yet implement proposal v2's replay, three agents,
  GRU, win model, market APIs, risk layer, reviewer, rule notebook or gate.
- Generated code runs through `exec`; that is acceptable for a local course
  prototype but unsafe for a public service.
- `download_season.py` needs a small real-data smoke test before a complete
  download.
- `nba_api` does not provide contracts, news or market prices.
- Prototype labels are generated from templates and are not a substitute for
  the proposal's held-out chronological evaluation.

## Submission target

Deliverables are the source code with sample replay days and setup
instructions, a 5–8 page report, a 10-minute presentation including the
complete demo, a backup recording, contribution statement, and LLM-usage
statement. The project deadline is **10 October 2026**.
