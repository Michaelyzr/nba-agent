# Market-Graded Learning for NBA Game-Impact Intelligence

DASC7606C Group 12, Track 2 (Agentic Framework Design). A multi-channel agent
system that turns late game news and NBA statistics into cited impact briefs,
grades them against prediction-market prices, and can trade under
code-enforced guardrails tailored by user type.

**Deadline:** code, demo, backup recording and report by **Saturday 10 October 2026**.

> **Project status (4 Oct):** this repository contains the earlier
> single-agent cited-stat prototype only. The replay, the three agents, the
> forecast models, the risk layer and the rule gate described below are the
> proposal's target and are not built yet. Do not present planned components
> as results.

## Contents

1. [Product and objectives](#1-product-and-objectives)
2. [System design](#2-system-design)
3. [Team, roles and work packages](#3-team-roles-and-work-packages)
4. [Schedule](#4-schedule)
5. [Handoffs between subgroups](#5-handoffs-between-subgroups)
6. [Data, models and evaluation](#6-data-models-and-evaluation)
7. [Demo script](#7-demo-script)
8. [Safety, risks and grading](#8-safety-risks-and-grading)
9. [How we work in this repo](#9-how-we-work-in-this-repo)
10. [Current prototype](#10-current-prototype)

## 1. Product and objectives

Within minutes of a change that rewrites a night (injury or rest, minutes
restriction, role or lineup shift, trade, back-to-back, "trending toward
playing"), the system produces a cited brief and calibrated forecasts of
minutes, points and win probability. One forecast core serves several
channels. Channels differ in the risk they may take, not in whether they see
the forecast.

| Channel | Party | Need | What they get, and their risk exposure |
| --- | --- | --- | --- |
| Primary customer (pays) | Fantasy and sports-media platforms | Lock-time projection refresh they can audit | Distributions, win-probability change, cited brief, freshness fields |
| User (may license) | Broadcasters and writers | Deadline-safe "who gains, who loses" copy | Cited narrative; no stake or betting language |
| User (enterprise) | Team coaching and analytics | Opponent rotation and matchup odds | Internal brief only; no public market signal |
| End user | Retail bettors and prediction-market users | How late news moves probabilities; optional guarded trade | Model vs market view; per-order confirm, caps, cool-downs, no parlays, no "lock" copy |

**Objectives**

1. **Serve by channel:** one forecast core; channel-specific briefs and risk policies.
2. **Inform with uncertainty:** who gains or loses minutes and shots, and how win
   probability moves, with ranges and sources.
3. **Prove with markets:** replay 2025–26 day by day and grade forecasts against
   recorded Kalshi or Polymarket prices; report flat or negative results honestly.
4. **Learn safely:** keep only gate-approved rules; show the learning agent beats
   no-learning and no-agent baselines on held-out months.
5. **Demo one night across channels**, including a guarded fill, a reused learned
   rule, and planted failures that get rejected.

**Why an agent:** it chooses which stats to pull, which forecasts to recompute,
which brief and risk policy fit the channel, and which past rules apply.
**Why not ChatGPT:** no calibrated distributions, no channel risk limits, no
learning gate, no QA against settled outcomes. ChatGPT is still run as a
baseline on the same inputs.

## 2. System design

```text
News + NBA stats (as of decision time)
            |
            v
   Forecaster (LLM agent) ----- calls forecast models
            |
            v
   Grader / trader (LLM agent)
            |
            v
   Risk limits (code) --> paper or live fill
            |
            v
      Game settles
            |
            v
   Reviewer (LLM agent): cause of each loss, proposed rule
            |
            v
   Gate (code): back-test on earlier days only
      | helps                | does not help
      v                      v
   Rule notebook          rejected, reason logged
      |
      +--> read by the forecaster and the trader on later days
```

| Part | Type | Job |
| --- | --- | --- |
| Forecaster | LLM agent | Read late news, query NBA stats, apply notebook rules, call the models, write channel-specific briefs |
| Market grader / trader | LLM agent | Compare model probability with market price; propose a paper or live trade with a cited reason |
| Reviewer | LLM agent | After settlement, label each failed or missed trade: news misread, minutes shared wrong, model wrong, or market already priced it; propose a rule |
| Risk limits | Code | Caps per trade, game and day; drawdown kill switch; no post-tip orders; ask-plus-fees fills; liquidity caps; retail confirm and cool-downs; no parlays; team and media cannot order |
| Gate | Code | Back-test each proposed rule on earlier days only; keep it if calibration or closing-line value improves by a set threshold; can freeze live trading |
| Replay and scorer | Code | Chronological replay on recorded prices; calibration, P&L, drawdown, closing-line value; flag policy breaches |

The agent that writes a rule never approves it. Risk limits are code, so an
agent cannot override them. All three agents call chat models through one
wrapper so a supplier (Gemini, OpenAI, Anthropic, DeepSeek) can be swapped
without changing agent code; every decision logs the supplier and model version.

A rule has a condition code can check and an action code can apply:

```json
{"when": {"team": "Y", "ruled_out_position": "C", "hours_to_tip_max": 2},
 "do": {"minutes_share": {"backup_C": 0.70, "others": 0.30}},
 "evidence": "helped on 14 of 18 past cases", "status": "active"}
```

Trader rules work the same way, for example "skip player-prop trades when the
news is older than 20 minutes" or "pause live size after three losing sessions".

`checks` rejects: leakage (inputs published after the decision), invalid
minutes, orders over channel limits, missing cited reason, retail "lock" or
"guaranteed" copy, parlays, live sends without risk clearance or retail
confirm, and rules the gate did not approve.

### Repository layout

Application code lives in the installable `nba_agent` package. Generated data,
trained artifacts and run outputs stay at the repository root and remain
git-ignored. New target-system modules should be added inside the matching
package instead of creating more top-level Python files.

```text
src/nba_agent/
  agents/       graph, nodes, LLM wrapper, forecaster and reviewer
  data/         table access, downloaders and sample-data builders
  domain/       typed shared contracts
  evaluation/   metrics, exports and ablations
  forecast/     baselines, trained models and forecast API
  policy/       validation, leakage checks, risk and rule gate
  replay/       chronological replay, fills and settlement
  rules/        versioned, gate-approved rule notebook
  skills/       prompt and calculation instructions
  ui/           Streamlit demo
tests/          unit, leakage, risk and end-to-end tests
data/sample/    small committed replay fixtures
models/         generated model artifacts (git-ignored)
runs/           generated replay outputs (git-ignored)
```

## 3. Team, roles and work packages

Subgroups are from the proposal and are **tentative**. Owners within a
subgroup are suggestions so every package has one name on it; swap freely, but
keep one owner per package for the contribution statement.

| Subgroup | People | Scope |
| --- | --- | --- |
| Data and replay | Wu Yaqi, Wang Yisong | Seasons, late-news feed, contract prices, `replay.py`, leakage test |
| Models | Zhou Yuanxiang, Li Yifei, Ngan Tsz Sui | Play classifier, GRU distribution model, win model, baselines |
| Agents | Yang Qianlang, Li Lanqiao, Fu Yuxuan | Forecaster, grader/trader, reviewer, gate, risk limits, checks |
| Evaluation and product | Shen Shangyi, Zhang Muqun | Scorer, ablations, Streamlit demo, recording, report |

### Data and replay

| ID | Work | Done when | Suggested owner | Due |
| --- | --- | --- | --- | --- |
| D1 | Freeze NBA stats 2022–23 to 2025–26 from `nba_api` (smoke-test `download_season.py --limit 5` first) | Parquet tables in `data/frozen/` load with the schemas in section 5 | Wu Yaqi | Mon 5 Oct |
| D2 | Late-news table with publish times from NBA injury-report PDFs; inactive list as fallback, stamped 30 minutes before tip | Every row has `published_at`, source and URL | Wu Yaqi | Mon 5 Oct |
| D3 | Kalshi 1-minute price history for game-winner and player-points markets; Polymarket as backup. **Check today how far back prop history goes** | Price table covers the test period; prop coverage dates written in this README | Wang Yisong | depth check Sun 4 Oct; data Mon 5 Oct |
| D4 | `replay.py`: day-by-day loop, as-of filtering, fill at ask plus fees, size capped by recorded volume, settlement | A plain baseline model trades a full month and settles | Wang Yisong | Mon 5 Oct |
| D5 | Leakage test: no input with a timestamp after the decision time | Test in `tests/` fails on a planted future row | Wu Yaqi | Tue 6 Oct |
| D6 | Commit a few small replay days to `data/sample/` so graders can run the demo without downloads | Demo runs from a fresh clone | Wang Yisong | Thu 8 Oct |

### Models

| ID | Work | Done when | Suggested owner | Due |
| --- | --- | --- | --- | --- |
| M1 | Baselines: 10-game rolling average and gradient boosting with teammates-out features | Both produce forecasts in the section 5 format; unblocks D4 | Ngan Tsz Sui | Mon 5 Oct |
| M2 | GRU over each player's last 20 games plus rest, teammates out and opponent pace, outputting a minutes and points distribution | Held-out calibration reported against M1 | Li Yifei | Tue 6 Oct |
| M3 | Play classifier: will the player play | Held-out Brier score reported | Zhou Yuanxiang | Tue 6 Oct |
| M4 | Win model: team ratings adjusted for who is out | Win probabilities per game, before and after a status change | Zhou Yuanxiang | Tue 6 Oct |
| M5 | `forecast/api.py`: one call that returns any model's distribution as of a timestamp, with overrides for "player X out" | Agents and replay call only this function | Ngan Tsz Sui | Tue 6 Oct |

### Agents

| ID | Work | Done when | Suggested owner | Due |
| --- | --- | --- | --- | --- |
| A1 | Multi-supplier LLM wrapper (extend the prototype `llm.py`) that logs supplier and model version per call | All agents call one wrapper | Yang Qianlang | Mon 5 Oct |
| A2 | Forecaster: tools over frozen data, applies notebook rules, calls M5, writes the three channel briefs | Brief cites news and stats; numbers come only from M5 | Yang Qianlang | Tue 6 Oct |
| A3 | Grader/trader and `policy/risk.py` | Over-cap, post-tip, parlay and unconfirmed retail orders are blocked | Li Lanqiao | Tue 6 Oct |
| A4 | Reviewer, `policy/gate.py` and the rule notebook | A proposed rule is accepted or rejected by back-test on earlier days only | Fu Yuxuan | Tue 6 Oct |
| A5 | `policy/checks.py` and planted-failure tests | Every check in section 2 has a failing test case | Fu Yuxuan | Tue 6 Oct |
| A6 | LangGraph loop wiring forecaster, trader, risk, settlement, reviewer and gate into the replay | Full loop runs over the development period and learns at least one rule | Yang Qianlang | Tue 6 Oct |

### Evaluation and product

| ID | Work | Done when | Suggested owner | Due |
| --- | --- | --- | --- | --- |
| E1 | `evaluation/scorer.py`: closing-line value, Brier score, P&L after fees, max drawdown, kill-switch trips, policy-test results | Scores the D4 baseline replay | Shen Shangyi | Mon 5 Oct |
| E2 | Hand-label 30 trades for reviewer blame accuracy | Labels in `evaluation/labels/`, made before seeing reviewer output | Zhang Muqun | Wed 7 Oct |
| E3 | Ablations on the frozen test period (section 6) and the headline chart | Table and chart regenerate from one command | Shen Shangyi | Thu 8 Oct |
| E4 | Streamlit demo of one night across channels (section 7) | Runs from `data/sample/` | Zhang Muqun | Thu 8 Oct |
| E5 | Backup recording of the full demo | Video file linked from the report | Zhang Muqun | Fri 9 Oct |
| E6 | Report assembly (each subgroup writes its own section), contribution statement, LLM usage statement | 5–8 page PDF | Zhang Muqun (lead), all | draft Fri 9, final Sat 10 Oct |

## 4. Schedule

The proposal planned: Thu 1 Oct data and replay done; Sun 4 Oct full agent loop
running; Tue 6 Oct freeze; Sat 10 Oct submit. The first two milestones were not
met, so the remaining days are re-planned below. The freeze stays on 6 October
because the test period needs frozen prompts, thresholds and models.

| Date | Checkpoint |
| --- | --- |
| Sun 4 Oct | Handoff formats in section 5 agreed; folders created; Kalshi prop-history depth checked; downloads started |
| Mon 5 Oct | **Checkpoint 1:** replay runs end to end with a baseline model trading; scorer reports closing-line value and Brier score |
| Tue 6 Oct | **Checkpoint 2 and freeze:** full agent loop runs on the development period and learns rules; prompts, thresholds and models frozen that night |
| Wed 7 Oct | Test-period runs and ablations start; reviewer labels done |
| Thu 8 Oct | Ablation table and headline chart; demo runs end to end; sample days committed |
| Fri 9 Oct | Report draft; backup recording; presentation dry run |
| Sat 10 Oct | Submit code, demo, recording and report |

**Cut order if behind:** ChatGPT sample, win model (props only), GRU replaced
by a small feedforward net, free-text news parsing (use inactive lists and
official statuses). **Never cut:** leakage test, risk limits, gate, the
no-learning and no-agent comparisons, backup recording.

## 5. Handoffs between subgroups

Agree these on day one so subgroups can build in parallel against fakes. Every
record carries a timestamp, and nothing may read data published after it.

**Data and replay → everyone** (`data/frozen/`, Parquet)

| Table | Key columns |
| --- | --- |
| `games` | `game_id`, `date`, `tip_time`, `home_team_id`, `away_team_id`, `home_pts`, `away_pts` |
| `player_games` | `game_id`, `player_id`, `team_id`, `min`, `pts`, `fga`, `fta`, `usage`, `started` |
| `news` | `news_id`, `published_at`, `game_id`, `player_id`, `status`, `source`, `url`, `text` |
| `prices` | `venue`, `market_ticker`, `game_id`, `player_id` (props), `line`, `ts`, `bid`, `ask`, `volume` |
| `settlements` | `market_ticker`, `settled_at`, `outcome` |

**Models → Agents and Evaluation** (returned by `forecast/api.py`)

```json
{"game_id": "...", "player_id": "...", "as_of": "2026-02-03T23:40:00Z",
 "model": "gru-v1", "target": "pts",
 "quantiles": {"p10": 14.2, "p50": 21.0, "p90": 28.5},
 "p_over": {"22.5": 0.41}, "p_play": 0.93,
 "overrides": {"out": ["player_id_x"]}}
```

**Agents → Replay and Evaluation** (one decision per proposed order)

```json
{"decision_id": "...", "as_of": "...", "channel": "retail",
 "market_ticker": "...", "side": "yes", "p_model": 0.58, "price": 0.51,
 "stake": 20, "reason": "cited text", "citations": ["news_id", "game_id"],
 "rules_applied": ["rule_id"], "risk_result": "approved", "llm": "supplier/model@version"}
```

**Reviewer → Gate → notebook:** the rule JSON in section 2, plus `proposed_at`,
`gate_result`, `backtest_days` and `metric_change`.

## 6. Data, models and evaluation

**Data**

| Data | Source |
| --- | --- |
| NBA statistics 2022–23 to 2025–26: game logs, advanced and tracking box scores, play-by-play, lineups, on/off | `nba_api`, frozen once locally; the agent queries it filtered to the decision time |
| Late game news with publish times | NBA.com injury-report PDFs and team or beat notes; fallback is each game's inactive list, treated as news 30 minutes before tip |
| Contract prices, 1-minute history | Kalshi historical API (game winner, player points); Polymarket price history as backup |
| Gaps in price or prop history | Third-party archives such as ScoreTape or CryptoStruct, bought only for the days needed |

**Deep learning:** play classifier; GRU minutes and points distribution;
win model; baselines of a 10-game rolling average and gradient boosting. The
GRU output is a distribution, so "probability of over 22.5 points" is computed
by code, not by the LLM.

**Evaluation periods:** development October to January (tune prompts,
thresholds, gate); test February to April, frozen. If prop prices start in
March, props are evaluated on March to April only.

| Setup | What it isolates |
| --- | --- |
| Full agent | The product |
| Same agent, no learning | Value of the teaching loop |
| Plain model, no agent: trade whenever model and price differ by a threshold | Value of the agent |
| Fixed 60/40 minutes rule | Value of learned rules |
| ChatGPT given the same news, stats and prices (100-trade sample) | Why not ChatGPT |

**Metrics:** closing-line value (entry price vs price at tip-off; less noisy
than profit); profit after fees; Brier score against the market price; maximum
drawdown and kill-switch trips; reviewer blame accuracy on 30 hand-labelled
trades; policy tests (over-cap live order blocked, retail order without
confirm blocked, team channel cannot order).

**Headline chart:** cumulative closing-line value over the test period, with
and without learning.

## 7. Demo script

1. A replayed night: late news changes a starter's status. Platform and desk
   briefs update; retail sees model vs market. Under caps, the trader submits a
   paper or live fill with a cited reason.
2. The game settles. The reviewer explains a loss, proposes a rule, and the gate
   accepts it.
3. A later night uses the rule.
4. Planted failures are rejected: over-cap order, post-tip order, retail live
   order without confirm, "guaranteed lock" copy.
5. The headline chart and ablation table. A backup recording is ready.

## 8. Safety, risks and grading

**Exposure by channel:** team and media get intelligence without wagering.
Retail and platform trading get intelligence plus hard order limits the LLM
cannot bypass. Retail live mode needs explicit confirmation per order; no
silent auto-betting, no parlays, no guaranteed-edge language. Season results
use recorded prices; live orders are allowed only when risk code, venue rules
and (for retail) user confirm all clear. Flat or negative closing-line value
is reported and can tighten or pause live trading through the gate.

| Risk | Response |
| --- | --- |
| Prop price history is short | Evaluate props on covered months; game-winner markets cover the full test period |
| Market is too efficient to beat | Report a flat or negative result honestly; no false edge |
| Recorded prices ignore thin order books | Fill at the ask, cap size by recorded volume, charge fees |
| Rules overfit | Minimum case count, cap on active rules, retire rules that stop helping |
| Gambling harm | Tiered exposure, hard caps, kill switch, cool-downs, retail confirm, no parlays; report covers problem gambling |
| "Is this a betting bot?" | Same forecast, different risk policy per channel; live trading is a guarded mode, not the product |
| Venue or regional blocks | Respect exchange geo rules; fall back to paper replay |
| Six days left | Two checkpoints, freeze on 6 October, the cut list |

**How Track 2 is graded** (course brief, 100 points)

| Criterion | Points | What is assessed |
| --- | --- | --- |
| Problem and user need | 15 | Clear users and a meaningful problem |
| Tool design and usability | 15 | Useful features and a coherent user workflow |
| Deep learning approach | 20 | Appropriate models, justified choices, credible data plan |
| Implementation and demo | 25 | A functioning tool that shows the full workflow from input to output |
| Evaluation and limitations | 15 | Meaningful testing, evidence of usefulness, honest failures and risks |
| Report, presentation and Q&A | 10 | Clear explanation and well-supported answers |

The work must go beyond a basic model or API call. A live demo is required,
with a backup recording. Each member's score is the group score times a
contribution factor taken from the contribution statement and evidence, so
keep your work in your own commits and pull requests.

**Deliverables:** report (5–8 pages excluding references); 10-minute
presentation including the demo, plus 5 minutes of Q&A; source code with setup
instructions and sample inputs, no API keys or restricted data; contribution
statement; LLM usage statement.

## 9. How we work in this repo

- `main` is protected. Work on a branch named after your subgroup and package,
  for example `data/d4-replay`, `models/m2-gru`, `agents/a3-risk`,
  `eval/e1-scorer`, then open a pull request.
- Ask Michaelyzr for write access. Pull requests will need one approval once
  teammates are added.
- Never commit `.env`, API keys, raw downloads or trained weights. Keys go in
  `.env` (see `.env.example`); downloads go under git-ignored `data/` folders.
- Build against the section 5 formats. If you need to change one, say so in the
  group chat and update this README in the same pull request.
- Record any AI tools you used in your pull request description. The report's
  LLM usage statement is assembled from those notes.

## 10. Current prototype

The existing code is one LangGraph agent that answers cited questions over
synthetic NBA-shaped data (real team names, made-up players, stats, contracts
and news). Every number cites saved games, every quotation cites a saved
paragraph, and a signing estimate is labelled as a prediction with its
held-out error. Parts worth reusing for the target build: `llm.py` (A1),
the LangGraph patterns in `graph.py` (A6), the check-and-retry design in
`checks.py` (A5), and `download_season.py` (D1).

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

nba-agent-sample
python -m nba_agent.forecast.train_signing

# Optional cross-encoder fine-tunes; each downloads a base model
python -m nba_agent.forecast.train_ranker
python -m nba_agent.forecast.train_support

cp .env.example .env
```

Without `GEMINI_API_KEY`, the chat steps use offline rules and templates.

### Run

```bash
nba-agent "Did <player>'s true shooting change after he was traded this season?"
nba-agent --plant playoffs "<same question>"
streamlit run src/nba_agent/ui/app.py
nba-agent-evaluate
pytest -q
nba-agent-export
```

Supported questions and player names are generated in `data/eval/questions.json`.

| Package | Current role |
| --- | --- |
| `nba_agent.agents` | Single-agent LangGraph state, routing, retries, LLM access and prototype nodes |
| `nba_agent.data` | Cached table access, `nba_api` download and synthetic sample builder |
| `nba_agent.forecast` | Prototype ranker, support model, signing network and training utilities |
| `nba_agent.policy` | Deterministic checks for statistics, citations and quotations |
| `nba_agent.evaluation` | Prototype held-out evaluation and CSV exports |
| `nba_agent.skills` | Question-specific table and calculation instructions |
| `nba_agent.ui` | Streamlit prototype |
| `nba_agent.domain`, `nba_agent.replay` | Reserved boundaries for the target typed contracts and replay engine |

`--plant` faults (first attempt only): `playoffs`, `wrong_team`, `no_count`,
`rate_mix`, `invented_quote`, `wrong_paragraph`, `no_error`, `note_number`.

**Prototype limits:** generated code runs through `exec`, which is fine for a
local demo but unsafe for a public service; `nba_api` has no contracts, news or
market prices; prototype labels come from templates and do not replace the
chronological held-out evaluation above.
