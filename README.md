# Market-Graded Learning for NBA Game-Impact Intelligence

DASC7606C Group 12, Track 2 (Agentic Framework Design). A multi-channel agent
system that turns late game news and NBA statistics into cited impact briefs,
grades them against prediction-market prices, and can trade under
code-enforced guardrails tailored by user type.

**Deadline:** code, demo, backup recording and report by **Saturday 10 October 2026**.

> **Project status (5 Oct):** everything runs end to end on real data:
> ESPN box scores and inactive lists for three seasons, and Kalshi game-winner
> prices for 2025-26 (1,316 games). M1–M5 are trained (`forecast/`). The
> agent loop is market-anchored and learns gated notebook rules. Ablations,
> calibration and policy tests are in `evaluation/results/`, and the Streamlit
> demo is `app.py`. Results are summarised in [Results](#results). By default
> the agent runs with offline rules; add a Gemini key and `--llm` for the LLM
> steps.

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
learning gate, no QA against settled outcomes. The plain-LLM / ChatGPT-style
baseline is arm B of the LLM tool-agent evaluation
(`docs/preregistration_llm_agent.md`, `evaluation/llm_agent_eval.py`,
`agents/tool_agent.py`): same news, stats and prices, one Gemini call, same
trading rule as the deterministic agent.

## 2. System design

The trading agent is one LangGraph (`agents/graph.py`) with two phases.
**Decide** runs at every news item in the six hours before tip-off, and once
an hour before tip. **Review** runs after each replay day. Replay hands the
graph an as-of view: only news, finished games and quotes published by that
moment. Implemented in `agents/graph.py`, `agents/notebook.py`, `llm.py` and
`replay.py`.

Blue = LLM judgment; green = forecast numbers; grey = code; yellow = what the
user sees or what is logged. Dotted lines are gate-approved rules feeding the
next night's decisions.

```mermaid
flowchart TD
    A["New injury report or news item<br/>(or 1 h before tip-off)"] --> B["Trigger + as-of view<br/>only data published so far"]

    B --> C["News investigator agent<br/>What does the news mean?<br/>Which players and markets are affected?"]
    N[("Rule notebook<br/>gate-approved rules")] -.-> C
    C --> D["Forecast models via forecast()<br/>play chance · minutes · points · win %<br/>before vs after the news"]

    D --> E["Market analyst agent<br/>Model vs market price<br/>How far has the price already moved?"]
    N -.-> E
    E --> F{"Gap left<br/>after fees?"}
    F -- No --> G["Brief: already priced in,<br/>no action"]
    F -- Yes --> H["Brief + proposed order<br/>with cited reason"]

    G --> I{"Checks pass?<br/>citations before decision,<br/>numbers match models,<br/>no lock wording"}
    H --> I
    I -- "No (one retry)" --> C
    I -- Yes --> J{"Risk limits pass?<br/>caps · before tip-off ·<br/>channel · no parlays"}
    J -- No --> K["Order blocked, reason logged<br/>(brief still delivered)"]
    J -- Yes --> L["User sees brief<br/>and confirms order"]

    L --> M["Game settles<br/>profit, closing-line value, calibration"]
    K --> M
    M --> O["Reviewer agent<br/>Why did we lose or miss?<br/>Propose a rule"]
    O --> P{"Gate: does the rule help<br/>on earlier days only?"}
    P -- Yes --> N
    P -- No --> Q["Rule rejected, reason logged"]

    classDef llm fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef code fill:#f1f5f9,stroke:#475569,color:#0f172a
    classDef dl fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef out fill:#fef9c3,stroke:#ca8a04,color:#713f12
    class C,E,O llm
    class B,I,J,M,P,F code
    class D dl
    class G,H,K,L,Q out
```

| Diagram box | Graph node | Type | Job |
| --- | --- | --- | --- |
| News investigator (Forecaster in the work packages) | `investigate` | LLM | Read late news, apply notebook rules, name who is out and which markets are affected |
| Forecast models | `forecast` | Code / DL | `forecast/api.py` before vs after the news: play chance (M3), minutes and points (M2 GRU), win % (M4) |
| Market analyst (Grader / trader) | `analyse` → `propose` / `no_action` | LLM + code | Code computes the gap after fees; the LLM may drop a trade and explain, never add one |
| Checks | `checks` | Code | Citations must predate the decision; numbers must match the models; no “lock” wording; one retry then block |
| Risk limits | `risk` | Code | Caps, no post-tip orders, channel permissions; team and media cannot order; no parlays |
| Confirm / blocked | `confirm` / `blocked` → `deliver` | Code | Brief always delivered; replay auto-confirms; live demo waits on the user |
| Reviewer | `review` | LLM | After settlement, blame the loss or miss and propose a machine-checkable rule |
| Gate + notebook | `gate` → `save_rule` / `reject_rule` | Code | Back-test on earlier days only; only the gate sets a rule active |
| Replay and scorer | `replay.py` | Code | Decision times, fills at ask plus fees, P&L, closing-line value |

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

### Target repository layout

New code goes into these folders. `models/` is reserved for trained artifacts and
is git-ignored, so model code lives in `forecast/`.

```text
data_sources/   espn.py, kalshi.py, sample.py                      Data and replay
                (nba_stats.py, news.py, polymarket.py: same formats, for unblocked networks)
replay.py       chronological day loop, fills, settlement          Data and replay
forecast/       history.py, baselines.py, gru.py, play.py, win.py, api.py, train.py   Models
                (dev_data.py: synthetic season for development)
agents/         graph.py, notebook.py, briefs.py, demo_data.py     Agents
                (risk limits, checks and gate live in graph.py and notebook.py)
llm.py          chat-model wrapper with offline fallback           Agents
evaluation/     scorer.py, ablations.py, results/                  Evaluation and product
app.py          Streamlit demo                                     Evaluation and product
tests/          leakage, risk, model and planted-failure tests     everyone
data/sample/    committed history and a few replay nights          Data and replay
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
| D1 | Freeze NBA stats 2022–23 to 2025–26 from `nba_api` (smoke-test `python -m data_sources.nba_stats --seasons 2025-26 --limit 5` first) | Parquet tables in `data/frozen/` load with the schemas in section 5 | Wu Yaqi | Mon 5 Oct |
| D2 | Late-news table with publish times from NBA injury-report PDFs; inactive list as fallback, stamped 30 minutes before tip | Every row has `published_at`, source and URL | Wu Yaqi | Mon 5 Oct |
| D3 | Kalshi 1-minute price history for game-winner and player-points markets; Polymarket as backup. Depth checked 4 Oct: game winner (`KXNBAGAME`) covers the whole 2025–26 season, 2,898 markets from October; player points (`KXNBAPTS`) start 19 Nov 2025, 23,562 markets, so props cover the full February–April test period | Price table covers the test period | Wang Yisong | data Mon 5 Oct |
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
| A1 | Multi-supplier LLM wrapper (extend `llm.py`) that logs supplier and model version per call | All agents call one wrapper | Yang Qianlang | Mon 5 Oct |
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
| `games` | `game_id`, `date` (Eastern), `tip_time`, `final_at` (tip + 3 h; box score visible from then), `home_team_id`, `away_team_id`, `home_team`, `away_team` (tricodes), `home_pts`, `away_pts` |
| `player_games` | `game_id`, `player_id`, `team_id`, `min`, `pts`, `fga`, `fta`, `usage`, `started` |
| `players` | `player_id`, `player_name` |
| `news` | `news_id`, `published_at`, `game_id`, `player_id`, `status`, `source`, `url`, `text` |
| `markets` | `venue`, `market_ticker`, `kind` (`game` or `pts`), `game_id`, `team` (yes side), `player_id` and `line` (props), `title` |
| `prices` | `venue`, `market_ticker`, `ts` (end of the 1-minute candle), `bid`, `ask`, `volume` |
| `settlements` | `market_ticker`, `settled_at`, `outcome` (1 if yes) |

The schemas are also in `data_sources/__init__.py` (`SCHEMAS`); `write_table`
refuses a table that is missing a column. All times are UTC.

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
| Plain LLM / ChatGPT-style baseline (LLM tool-agent arm B; see below) | Why not ChatGPT |

**Metrics:** closing-line value (entry price vs price at tip-off; less noisy
than profit); profit after fees; Brier score against the market price; maximum
drawdown and kill-switch trips (enforced; see gate audit below); reviewer
blame accuracy on 30 hand-labelled trades (sheet sampled, **pending
annotation** — `evaluation/labels/reviewer_label_sheet.csv`, kappa via
`evaluation/reviewer_agreement.py`); policy tests (over-cap live order
blocked, retail order without confirm blocked, team channel cannot order).

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

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add GEMINI_API_KEY for the LLM steps (optional)
```

Without `GEMINI_API_KEY`, the agent's LLM steps use offline rules.

### Quick start from a fresh clone (no downloads)

`data/sample/` holds every game, box score and inactive list for 2023-24 to
2025-26 plus the markets and prices of a few test-period nights.

```bash
python -m forecast.train --source sample          # trains M1-M4 into models/ (about 3 minutes)
streamlit run app.py                              # the demo
python -m agents.graph --source sample --start <night> --end <night> --name sample-night
```

### Building the frozen data

Run from the repo root, in this order (each step caches its downloads under
`data/raw/`, so rerun to resume):

```bash
python -m data_sources.espn                                     # games, box scores, inactive lists (2023-24 to 2025-26)
python -m data_sources.kalshi download --start 2025-10-21 --end 2026-06-30 --series KXNBAGAME
python -m data_sources.sample                                   # refresh data/sample/ from data/frozen/
```

**Which NBA source.** stats.nba.com and the NBA injury-report PDFs block cloud
and VPN addresses (time-outs and 403 errors), so the frozen tables come from
ESPN's public JSON (`data_sources/espn.py`). It has every box score and each
game's inactive list with the reason (injury, illness, rest, suspension).
Following section 6, inactive-list entries become news stamped 30 minutes
before tip-off, so the agent sees no news earlier than that. Real injury
reports come out hours earlier, which makes this a conservative test. On an
unblocked network, `data_sources.nba_stats` and `data_sources.news` give
nba_api box scores and timestamped injury reports in the same formats.

### Training the models

```bash
python -m forecast.train                         # M1-M4 on games before 2026-02-01; held-out tables in evaluation/results/
```

M1 baselines (`forecast/baselines.py`), M2 GRU (`forecast/gru.py`), M3 play
classifier (`forecast/play.py`) and M4 win model (`forecast/win.py`) share one
leakage-safe history index (`forecast/history.py`): every feature for a game
uses only games that tipped off before it. `forecast/api.py` (M5) is the one
entry point the agent, scorer and demo call.

A separate, files-only experiment on player features and probability
calibration lives in `forecast/player_experiment/`; see
[its README](forecast/player_experiment/README.md). It does not change M1-M5.

### M6: market-impact model (learning from the market's reaction)

```bash
python -m forecast.impact                                  # dataset, ImpactNet, baselines; models/m6/impact.pkl, m6_heldout.csv, m6_vs_baselines.png
python -m agents.graph --signal impact --start 2026-02-01 --end 2026-04-12 --name m6-test --no-learn
python -m evaluation.m6_ablation                           # M6 agent vs the other setups; runs/m6/, m6_ablation.csv and .md
```

- **What it predicts.** Where the home game-winner price will be at tip-off:
  the move of the home mid from the decision time to the mid at tip. The
  market grades the model directly (the label is the closing price), so no
  game result is needed and every decision time is a training row.
- **Rows.** One per (game, time): the replay's own decision times (news inside
  the 6-hour window, tip − 60 min) plus a grid every 30 minutes from tip − 6 h to
  tip − 30 min, and tip − 15 min. Rows need a quote under 5 minutes old, as
  the agent does. Train on games before 15 Jan 2026 (7,826 rows, 602 games),
  validate on 15–31 Jan (1,638 rows), test 1 Feb – 12 Apr (6,508 rows, 501
  games, 771 at real decision times). Play-in and playoff games after 12 Apr
  are a holdout scored once (1,127 rows, 87 games).
- **Inputs, as-of only.** The home market's last 6 hours as 24 steps of 15
  minutes (mid relative to now, spread, volume, a has-quote mask); the current
  mid, spread and last-hour volume; the mid 24 h before tip and the move
  since; hours to tip and minutes since the latest news; M4 before and after
  the news and the shift; and players out, missing minutes and missing points
  per team (M4's rotation, `forecast/win.py` `team_side`). M4 is the saved
  model trained on games before 1 Feb, so it is in-sample on M6's Nov–Jan
  training rows. `tests/test_impact.py` checks that every feature is unchanged
  when all quotes and news after the decision time are deleted or replaced.
- **Architecture.** `ImpactNet`: a GRU (16 units) over the price sequence,
  concatenated with the 18 standardised static features, then an MLP to the
  10th, 50th and 90th percentiles of the move. The quantiles are kept ordered
  with softplus increments and trained with pinball loss. Fixed seed, Adam
  with weight decay, early stopping on validation (best epoch 28).
- **Baselines on the same rows.** Zero move (the market is a martingale), linear
  regression and gradient boosting on the static features, and a fitted
  coefficient × M4's news shift. The fitted coefficient is +0.025: after the
  inactive list, the price moves on average 2.5% of what M4 says it should.

Held-out move to tip, in cents (`evaluation/results/m6_heldout.csv`). R² is
out-of-sample against zero move, so above 0 beats it.

| Rows | Model | MAE | RMSE | R² vs zero move | Correlation |
| --- | --- | --- | --- | --- | --- |
| Test, decision times (771) | Zero move | **0.833** | 1.489 | 0 | — |
| | M6 ImpactNet | 0.867 | 1.493 | −0.006 | 0.03 |
| | M4 shift only | 0.843 | **1.485** | +0.006 | 0.08 |
| | Linear | 0.937 | 1.492 | −0.004 | 0.11 |
| | Gradient boosting | 1.056 | 1.617 | −0.179 | 0.04 |
| Test, all rows (6,508) | Zero move | **1.346** | 2.481 | 0 | — |
| | M6 ImpactNet | 1.374 | **2.475** | +0.004 | 0.07 |
| Holdout, decision times (94) | Zero move | **0.660** | 1.149 | 0 | — |
| | M6 ImpactNet | 0.717 | 1.147 | +0.003 | 0.08 |

![M6 against the baselines](evaluation/results/m6_vs_baselines.png)

- **Trading.** `MarketAgent(impact=load_impact())`, or `--signal impact`,
  prices the home market at p = current mid + M6's median move, and the away
  market at 1 − p. It keeps the fee-adjusted edge threshold, checks, risk
  limits, notebook rules and one position per game. Results are in
  `evaluation/results/m6_ablation.md`, with day-clustered bootstrap CIs from
  `evaluation/stats.py`.
  - The M6 agent made **0 trades** on the test period and on the holdout,
    with or without learning, so it is identical to never trading ($0).
  - M6's median predictions stay within ±1.25¢. To trade, a prediction must
    beat half the spread plus the fee plus the 4¢ edge, about 5.5¢.
  - As a diagnostic only, M6 with the threshold off takes its preferred side
    in every game: 501 test trades, mean CLV −0.0048 (95% CI −0.0064 to
    −0.0035), CLV −$134, P&L −$652 (CI −$1,642 to +$268). On the holdout:
    87 trades, CLV −0.0040, P&L +$109 (CI −$250 to +$510). A CLV of about
    −0.5¢ per contract is roughly half the spread, so M6's direction adds
    nothing beyond what crossing the spread costs.
- **Interpretation.** The zero-move baseline wins. No model predicts the move
  to tip better than "the price will not move" on MAE: every model's excess
  MAE has a day-clustered 95% CI above zero (see the chart). The best R² is
  under 1%, which is too small to matter.
  In this window (6 hours to 15 minutes before tip), Kalshi's NBA game-winner
  price behaves like a martingale with respect to everything we can see:
  its own recent path, the clock, M4 and the inactive list. M6 learned this
  and predicts moves near zero, so the agent sensibly stays out. This is a
  null result, not an edge.
- **Caveats.**
  - Small sample: 501 test games and 87 holdout games, and rows from the same
    game are strongly correlated.
  - Our news is the inactive list stamped 30 minutes before tip. Real injury
    reports arrive hours earlier, so the market may react before our first
    news row.
  - M4 is in-sample on M6's training rows.
  - Labels use the last quote at or before tip, as the replay's CLV does.

### Running the agent loop

#### Pregame news polling and fair odds

`agents/pregame.py` adds a separate pregame-only LangGraph:
**historical baseline → retrieve news → extract factors → reforecast → record**.
The runner repeats this graph at a fixed interval, ending strictly before
tip-off. It computes both teams' probabilities and **fair decimal odds
(`1 / probability`, no bookmaker margin)** without requiring market quotes,
submitting orders or waiting for settlement.

The baseline uses only box scores final at its timestamp. Its history stays
fixed for that run, so subsequent changes reflect news factors. The default
is trained M4 via `forecast/api.py`; `--forecaster record` is an explicit
development baseline that needs no trained weights.

```bash
# Find a game id in the committed sample schedule.
python -c "import pandas as pd; print(pd.read_parquet('data/sample/games.parquet')[['game_id','date','away_team','home_team']].tail(10).to_string(index=False))"

# Offline: poll the frozen news every minute over six pregame hours.
python -m agents.pregame --mode replay --source sample --game-id <id> --forecaster record --name pregame-demo

# Concrete example: ORL @ BOS on 2026-04-12, with late inactive news.
python -m agents.pregame --mode replay --source sample --game-id 401811041 --forecaster record --name pregame-bos

# Trained M4 (train it first with python -m forecast.win --source sample).
python -m agents.pregame --mode replay --source sample --game-id <id> --name pregame-model

# Live: requires a FUTURE game in data/frozen/games.parquet, plus historical
# player_games.parquet and players.parquet with matching player ids.
python -m agents.pregame --mode live --source frozen --game-id <future-id> --poll-seconds 60 --name pregame-live

# One poll, or a shorter window (timezone-aware timestamps required).
python -m agents.pregame --mode live --source frozen --game-id <future-id> --once --name pregame-once
python -m agents.pregame --mode replay --source sample --game-id <id> --forecaster record --start-time <ISO-time-with-zone> --until <ISO-time-with-zone> --name pregame-window
```

Live sources are ESPN and CBS Sports NBA RSS, the latest matching NBA injury
report, Shams Charania's X posts, and **both teams' official and reporter X
accounts**. [The source catalog](docs/news_sources.md) lists all 30 teams,
account links, affiliations and review evidence. Three verified team PR accounts
are also included. Only the two playing teams' accounts are queried per game;
other teams' posts do not enter the source plan.

X uses the [official recent-search API](https://docs.x.com/x-api/posts/search/quickstart/recent-search).
Set `X_BEARER_TOKEN` in the local environment or `.env` (see `.env.example`);
the token needs recent-search access. No token means **unconfigured**, not a
successful search with no news. Other sources keep running. The API searches
at most seven days, uses a 30-second indexing allowance and a five-minute
overlap, and persists unfinished pagination across polls/restarts. Rate limits
defer retries without losing the pending window. Posts from the final indexing
window may not become searchable before tip-off.

```bash
# Inspect a game's planned accounts without network access or credentials.
python -m data_sources.news_registry --teams BOS LAL
```

`--news-registry <json>` replaces the source catalog. `--rss-url <url>` replaces
the default media feeds (repeat for multiple RSS feeds); unregistered feeds
have no authoritative priority. `--no-official` disables NBA PDF requests;
`--no-x` explicitly disables X requests.
Replay reads only the supplied table; it never searches today's web for a
historical game. Live mode ignores retrospective inactive-list news.
The local sample contains historical games, so it cannot be used as a current
live schedule. Schedule/roster refresh is currently the data pipeline's job.

Factors currently supported:

| Factor | Model input |
| --- | --- |
| Out / available | Lost share 1 / 0; source priority resolves conflicting reports |
| Doubtful / questionable / probable | Lost share 0.75 / 0.50 / 0.25 |
| Explicit minutes limit | Expected minutes capped against the player's recent usual minutes |
| Ambiguous news, rumours, trades or role changes without a supported status | Logged as review-only, no invented numerical adjustment |

Uncertain-status weights are **scenario assumptions awaiting calibration**,
not outputs from a trained participation classifier. M4 uses the expected
missing minutes and points as features; this extension likewise needs
held-out calibration. A minutes limit and uncertain participation combine
as `1 - P(play) × allowed_minutes / usual_minutes`. Repeated news does not
compound a player's impact. The conflict order is **NBA official injury report
→ team official/PR account → ESPN/CBS → Shams → team reporter → unclassified**.
Within a tier, the latest publication wins. Each source's latest evidence is
kept: a newer contradictory personal post stays pending until a primary source
changes its report; it does not silently override an older authoritative report.
With no primary report, Shams/reporter factors may update the forecast with
`secondary_only` confirmation. All contradictory evidence remains in snapshots.
RSS parsing requires a full player name from the as-of
roster, explicit status language, and "today/tonight" on the game's Eastern
date; multi-player or ambiguous sentences remain review-only. X also supports
unique roster surnames and explicit injury-list headings on the game's Eastern
date. Images, unsupported lineup/role changes and ambiguous text retain their
original post/link for review; image OCR is not implemented. A selected
available status clears an absence; an explicit minutes cap remains until
replaced by a later cap.

Every live item records publication time, first observation time, source and
URL; extracted player factors also record player id. Future items and observations are rejected. Requests have
timeouts; failed sources leave previous factors intact, mark the poll
degraded, and retry at the next interval. No new HTTP request starts at or
after tip-off; a request already in progress may finish after tip, in which
case its news is not used to emit another forecast.

`runs/<name>/` contains `snapshots.jsonl` (initial/current probabilities,
odds, changes, factors, conflicts, per-source coverage and errors), `news.jsonl`
(deduplicated player factors and application result), `evidence.jsonl` (original
articles/posts, including review-only items), and `state.json` (restart state,
source evidence and API pagination; no credentials). Reusing the same
name resumes the same baseline and factors; replay resumes at the next poll
unless an explicit start time is supplied. Configuration mismatches or time
travel require a new name. Model weights and historical tables should remain
fixed during a resumed run. The Streamlit **Pregame news loop** tab shows the
offline probability curve, decimal odds, applied evidence and per-game source
plan. The displayed source plan is configuration, not a live retrieval claim.

```bash
pytest -q tests/test_pregame.py tests/test_news_sources.py
```

#### Independent in-play news and fair odds

`agents/inplay.py` runs a separate post-tip loop over current scores, remaining
time, injury exits/returns, ejections and foul-outs. It uses NBA liveData with an
ESPN fallback, a dedicated X filtered stream for Shams and both teams' accounts,
and media RSS. Authoritative reports take priority over secondary posts.
The pregame runner, UI, state and model weights stay unchanged; in-play outputs
live under `runs/inplay/<name>/` and have their own read-only monitor.

```bash
# Synthetic event/score scenarios, not historical NBA play-by-play; no network.
python -m agents.inplay --mode replay --source sample --game-id 401811041 --initial-p-home 0.60 --name inplay-demo
streamlit run inplay_app.py

# Requires a current game in the local schedule and historical/player tables.
python -m agents.inplay --mode live --source frozen --game-id <current-id> --poll-seconds 5 --name inplay-live
```

Configure `INPLAY_X_BEARER_TOKEN` separately, with filtered-stream and recent-search
access. Five seconds is the target score polling period; publication, API and
request delays still apply. Stale or unavailable scores suppress fresh odds.
The default in-play model and injury weights are prototypes awaiting calibration;
near-real-time retrieval does not establish forecast accuracy. See
[the in-play guide](docs/inplay.md) for source priority, recovery, training,
replay schemas and independent state files.

#### Live Polymarket markets (read-only)

`streamlit run app.py` opens on the **NBA Polymarket Live Markets** page, which
shows current NBA markets from Polymarket's public API with no keys and no
trading. Use the sidebar **Page** switch for **Historical replay and agent**
(the tabs below). See [the Polymarket live guide](docs/polymarket_live.md).

#### Existing market replay and learning

`agents/graph.py` is the section 2 loop as one LangGraph with two phases.
At every replay decision time it runs trigger → news investigator → forecast
→ market analyst → checks (one retry) → risk → confirm or blocked. After each
replay day it runs settle → reviewer → gate → rule notebook.

```bash
python -m agents.graph --draw                                               # Mermaid of the compiled graph
python -m agents.graph --start 2026-02-01 --end 2026-02-28 --name agent-feb             # real data, trained models
python -m agents.graph --start 2026-02-01 --end 2026-02-28 --name agent-feb --no-learn  # ablation
python -m agents.graph --start 2026-02-01 --end 2026-02-28 --llm                        # with Gemini
python -m agents.graph --plant lock_wording --start 2026-02-10 --end 2026-02-10         # show a blocked order
python -m agents.graph --source synthetic --forecaster record --start 2026-01-01 --end 2026-01-31
```

- **LLM steps.** The news investigator, market analyst and reviewer use Gemini
  through `llm.py` when `--llm` is set and a key is in `.env`. Without one, they
  fall back to offline rules, so the loop runs with no network. An LLM can drop
  a candidate trade but never add one. Every number comes from code.
- **Forecasts.** `forecast/api.py` by default: M4 win probability before and
  after the news (rotation players ruled out). `--forecaster record` uses the
  old win-rate placeholder, kept as an ablation.
- **Briefs.** `agents/briefs.py` turns one forecast into the four channel
  briefs (platform, media, team, retail). Media and team briefs are checked for
  market language; retail briefs need a confirm per order.
- **Rules.** Default `--gate split`: the reviewer selects on the last 7 market
  days and the gate tests the 14 before them (disjoint), keeping a rule if
  total CLV dollars rise by ≥ $2 over at least 3 changed trades.
  `--gate legacy` is the published overlapping-window / mean-CLV rule.
  Expired rules can be renewed. `--kill-switch 100` stops new fills once the
  day's realised P&L (settled games only) is below −$100.
  `--sizing kelly --kelly-fraction 0.25` uses fee-aware fractional Kelly
  capped by the order / game / day limits.
- **Synthetic data.** `agents/demo_data.py` invents markets whose prices react
  10 minutes after injury news. It exists to develop the loop. Its P&L means
  nothing, so never report it.
- **Output.** `runs/<name>/` gets `decisions.parquet`, `fills.parquet`,
  `notebook.json` and `trace.jsonl` (every step of every decision).

### Evaluation and demo

```bash
python -m evaluation.ablations                   # section 6 ablations, calibration, policy tests, headline chart
python -m evaluation.m4_report                    # M4 win model vs the market: m4_vs_market.png and tables
python -m evaluation.trade_visuals                # how the agent trades: trade_flow.png, trade_example.png, trade_funnel.png
python -m evaluation.walkforward                  # walk-forward by month, play-off holdout, bootstrap CIs (~20 min)
python -m evaluation.walkforward --report-only    # re-score saved runs/walkforward and runs/holdout only
python -m evaluation.scorer runs/<name>          # score any run folder
python -m evaluation.scorer --policy-tests       # planted orders that must be blocked
streamlit run app.py                             # demo: live Polymarket page; replay page with replayed night, pregame loop, coach, league, briefs, learning, safety, models
```

### Coach: learning how the market works

The **Coach** tab (`agents/coach.py`) teaches users what is happening in a
game and how betting markets work. It runs on replayed nights with paper
money only.

1. **What's going on.** At a chosen decision time the coach explains the
   game in plain language. It covers the news so far, our model's win
   probability before and after it, and the market price now against 24 hours
   earlier. It then shows the anchored estimate and the break-even price after
   the spread and fee. Only information public at that moment is shown.
2. **Concepts in this game.** Lesson cards are triggered by the situation:
   - a price is a probability;
   - the spread and fee move your break-even;
   - how injury news moves a win probability;
   - whether the news is already priced in (chasing);
   - long shots look cheap;
   - the closing line is the scoreboard;
   - passing is a position.
3. **Your call.** The user backs a team or passes, with a paper stake. The
   order fills exactly as the agent's would: at the recorded ask plus fee,
   capped by traded volume. The rest of the night is then revealed: closing
   price, closing-line value, result and P&L. If the user passed, it shows
   what backing each team would have done. It also shows what the agent did
   at the same moment.
4. **Scoreboard and feedback.** Totals and habit tips over the session:
   paying above the closing price, long shots, chasing moved prices,
   negative-edge trades, and small samples being mostly luck.

The coach uses the same models, market anchor and fee formula as the agent,
so its explanations match the agent's decisions.
`tests/test_coach.py` checks that:
- its numbers match the as-of view;
- it never sees later news;
- paper fills match the replay;
- its text never uses promise words such as "lock" or "risk-free".

Every screen carries an educational, not-betting-advice notice.

### League: practising with play money

The **League** tab (`agents/league.py`) lets users practise on real past Kalshi
NBA markets with play money, ranked on skill rather than luck. Nothing is
deposited, bet or paid out.

- **A league** is a name plus a seeded slate of real decision points: 10 by
  default, one per game, from the test period (1 Feb – 12 Apr) or the play-offs.
  Every player sees the same slate. A game's decision time is one of its
  injury-news times, or tip − 60 min if it had no news.
- **Each decision** shows only as-of information, the same view the Coach
  uses: prices so far, the news, the model brief and the lesson cards. The
  player backs the home team, backs the away team or passes. Each player
  starts with a $1,000 bankroll; stakes are $5–$50 and never more than the
  remaining bankroll.
- **Fills** go through `coach.paper_trade`, so they work like the agent's:
  ask plus Kalshi fee, capped by volume, settled at the final result. After
  each call the closing price, closing-line value (CLV), result and P&L are
  revealed, along with what the house bots did at the same moment.
- **Leaderboard.** Ranked by mean CLV per contract, among players with at
  least 5 trades. P&L, CLV dollars, fees, pass rate and beat-the-close share
  are shown alongside. The "skill or luck?" badge uses the day-clustered
  bootstrap in `evaluation/stats.py`. It reads "skill" if the 95% CI of mean
  CLV is above 0, "costs" if it is below 0, and "too early to tell"
  otherwise.
- **House bots** play the same slate: "Never trade" ($0), "Agent" (the
  anchored `MarketAgent` with the sidebar's rule notebook, offline) and "Raw
  model" (the no-agent plain model). The personal summary compares the user
  with each bot on the decisions the user played, then adds the Coach's habit
  tips.
- **Storage.** One JSON file per league in `league_data/`, which is
  git-ignored.

```bash
streamlit run app.py      # open "League: practise with play money", enter a league name and a username
```

`tests/test_league.py` checks that:
- a decision never shows later prices, news or the score;
- fills and P&L match `paper_trade`;
- the leaderboard order, minimum trade count and badges are correct;
- the bots are included;
- stake caps and play order are enforced;
- leagues round-trip through JSON.

Results are written to `evaluation/results/` (committed, so the report and demo
use the same numbers). `python -m evaluation.workflow_diagram` redraws
`agent_workflow.png`, the LangGraph workflow figure.

### Results

Replay on real Kalshi prices. Development period: 1 Nov 2025 – 31 Jan 2026.
Test period: 1 Feb – 12 Apr 2026, with the same prices for every setup.
Stake $20 per order (caps: $50 per order, $100 per game, $300 per day). Fills are at the ask plus the Kalshi fee, capped by
traded volume. Closing-line value (CLV) is measured per contract against the
mid at tip. Offline rules; no LLM.

| Test-period setup | Trades | Mean CLV (s.e.) | P&L after fees | ROI | Max drawdown | Kill-switch trips |
| --- | --- | --- | --- | --- | --- | --- |
| Full agent: market anchor + learning | 74 | −0.0022 (0.0024) | −$32 | −2.2% | $195 | 0 |
| Agent: market anchor, no learning | 129 | −0.0035 (0.0015) | −$247 | −9.7% | $439 | 0 |
| Agent on win-rate placeholder model | 246 | −0.0038 (0.0009) | −$429 | −8.9% | $599 | 0 |
| Agent without market anchor (raw model) | 366 | −0.0047 (0.0007) | −$2,087 | −28.8% | $2,326 | 10 |
| Plain model, no agent | 366 | −0.0047 (0.0007) | −$2,087 | −28.8% | $2,326 | 10 |
| Agent on M6 market-impact signal (with or without learning) | 0 | — | $0 | 0% | $0 | 0 |
| Never trade | 0 | – | $0 | – | $0 | 0 |

![Cumulative CLV on the test period](evaluation/results/headline_clv.png)

"Kill-switch trips" in the table above counted days that lost more than $100
after the run (measured, not enforced). The gate-audit runs below enforce a
$100 daily realised-loss stop in `replay.py` and log every trip.

What the numbers say:

- **Our win model does not beat the market.** On 501 test games its Brier
  score is 0.186 (accuracy 73.9%). The market scores 0.164 one hour before
  tip and 0.163 at tip. Absences do help the model: ignoring them gives a
  Brier score of 0.192.
- **Trading the raw model loses heavily.** It buys underdogs it is
  underconfident about. Anchoring to the market changes this: the agent takes
  the market price 24 hours before tip and adds only the model's *news shift*
  (its probability after the news minus before). That cuts losses from
  −$2,087 to −$247.
- **Learning helps further on this window** (the walk-forward check below
  shows the P&L gain is not significant). The reviewer proposes one rule a day from the
  worst-CLV slice of settled trades. The gate keeps a rule only if a backtest
  on *earlier* days improves mean CLV. Over the test period it proposed
  37 rules and kept 5, for example "skip when buying a side priced at or
  below 35¢" and "skip when the market has already moved 2¢ or more against
  us". With learning, losses fall to −$32 on 74 trades, and the drawdown
  halves.
- **Honest caveats.**
  - Mean CLV is still slightly negative in every setup, and the learning
    agent's CLV is within about one standard error of zero. We avoid losing
    trades rather than finding an edge over the closing line.
  - The "development rules frozen" setup (`ablations.md`) is identical to "no
    learning": both rules kept during development expired (45-day limit)
    before the test period started.
- **Safety:** all 9 planted policy violations were blocked
  (`policy_tests.csv`).

**Research-process disclosure.** The market anchor and the data-driven
reviewer were designed *after* we looked at the first test-period results, so
1 Feb – 12 Apr is not a clean holdout, and the table above flatters the agent.
M4 was also trained on games before 1 Feb, so November–January is in-sample
for it. The walk-forward run and the play-off holdout below are the honest
checks.

#### Walk-forward and holdout, with confidence intervals

`python -m evaluation.walkforward` runs monthly windows over the 2025-26
season. Before each window, M4 (the win model, the only model that drives
trades) is retrained on every earlier game. M2/M3 stay as loaded. Each setup
is one replay over the season. The learning agent starts with an empty
notebook on 1 Nov and carries it forward, and its reviewer and gate only use
earlier days. The 95% confidence intervals (CI) come from a day-clustered bootstrap: game-days are
resampled with replacement, 2000 replicates, fixed seed. Never trade is $0
by construction. Full tables: `walkforward_summary.md`, `holdout.md`,
`significance.md`.

| Walk-forward, 1 Nov – 12 Apr | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] |
| --- | --- | --- | --- | --- |
| Full agent: anchor + learning | 157 | −0.0050 [−0.0079, −0.0018] | −$44 [−67, −21] | −$25 [−569, +564] |
| Agent: anchor, no learning | 276 | −0.0048 [−0.0068, −0.0027] | −$84 [−114, −54] | −$407 [−1,324, +553] |
| Raw model, no agent | 711 | −0.0046 [−0.0057, −0.0035] | −$255 [−326, −188] | −$2,475 [−3,995, −845] |
| Never trade | 0 | – | $0 | $0 |

![Walk-forward by month](evaluation/results/walkforward.png)

**Holdout: play-in and play-offs, 13 Apr – 14 Jun 2026** (87 games, 44
game-days). Nobody had looked at these games before this run. Each setup was
run once, with the default models. The full agent continued from the
test-period notebook.

| Holdout | Trades | Mean CLV [95% CI] | P&L after fees [95% CI] |
| --- | --- | --- | --- |
| Full agent: anchor + learning | 0 | – | $0 |
| Agent: anchor, no learning | 6 | +0.0050 [−0.0050, +0.0200] | −$74 [−176, +26] |
| Raw model, no agent | 59 | −0.0052 [−0.0083, −0.0026] | +$450 [−55, +1,171] |
| Never trade | 0 | – | $0 |

What the tests say:

- **No setup has positive closing-line value.** Over the walk-forward season,
  mean CLV is below zero for every setup, and every 95% CI excludes zero
  (one-sided p for CLV > 0 is 1.00 in all three). On the original test
  window, the learning agent's CLV cannot be told apart from zero
  (−0.0022, 95% CI −0.0067 to +0.0031). Orders fill at the ask, but CLV is
  measured against the mid, and the median half-spread is 0.005. So about
  −0.005 is what a trader with no edge would score: the agents pay the
  spread and show no edge over the closing price.
- **No setup beats never trading.** The full agent's walk-forward P&L is
  −$25 (95% CI −$569 to +$564; p = 0.53 for "better than never trading").
  The raw model loses significantly: −$2,475 (CI −$3,995 to −$845).
- **Learning helps by trading less, not by trading better.** Against the
  agent with no learning on the same days, the full agent loses $39 less CLV
  (CI +$18 to +$61, p = 0.001). Its mean CLV per trade is no better
  (−0.0002, CI −0.0019 to +0.0015). The P&L gain of +$382 is not significant
  (CI −$304 to +$1,047, p = 0.13).
- **The market anchor is the one clear effect.** Full agent minus raw model:
  P&L +$2,450 (CI +$854 to +$3,984, p = 0.002).
- **The holdout is too small to say much.** The anchored agent with no
  learning found only 6 trades in the play-offs. The learning agent's rules
  still active at that point (7¢ minimum edge, skip if the price has moved
  2¢ against us) filtered out all 6, so in the play-offs it equals never
  trading. The raw model's +$450 comes with significantly negative CLV
  (p = 0.999 for CLV > 0), so it is luck on outcomes, not an edge.
  Inactive-list news in the frozen data ends 7 May, so after that the agents
  trade on prices and team form only.

#### Gate audit, kill switch and Kelly

Pre-registered in `docs/preregistration_gate.md` before any of these runs.
Code: `--gate {legacy,split,split-edge}`, `--kill-switch 100`,
`--sizing kelly --kelly-fraction 0.25`. Safety invariants live in
`tests/test_safety_properties.py`. **Window (deadline):** 1 Feb – 12 Apr 2026
(learning arms warm up on Nov–Jan); the primary walk-forward season is
optional via `python -m evaluation.gate_audit --period wf`.

| Script | Output |
| --- | --- |
| `python -m evaluation.gate_audit` | `evaluation/results/gate_audit.{md,csv}` — legacy vs split gate, with/without the enforced kill switch, vs never trade |
| `python -m evaluation.gate_placebo` | `evaluation/results/gate_placebo.{md,csv}` — reviewer vs random templates / random slices under each gate |
| `python -m evaluation.kelly` | `evaluation/results/kelly.{md,csv}` — flat $20 vs ¼-Kelly (fee in the formula), vs never trade |

Headline findings (fill in from the result files after the run):

- **H1 (held-out gate days):** fewer rules should pass under `split` than under
  `legacy`. See pass rates and `rules_active` in the result files.
- **H2 (placebo):** under `legacy`, the reviewer's pass rate should be similar
  to random templates / slices (learning mainly reduces exposure).
- **H3 (kill switch):** enforced stop improves worst-day P&L without changing
  mean CLV (CI of the paired mean-CLV difference includes 0).
- **H4 (Kelly):** ¼-Kelly is not expected to beat flat $20 on P&L or CLV $;
  report as a null unless the P&L CI of Kelly − flat lies above 0.

**Reviewer labels (E2):** 30 losing trades sampled with seed 7606 from
`runs/results/test-full` into `evaluation/labels/reviewer_label_sheet.csv`.
Human label columns are empty — **pending annotation**. Do not fabricate
labels; Cohen's kappa is `python -m evaluation.reviewer_agreement` once both
annotators finish.

**ChatGPT / plain-LLM baseline:** see the LLM tool-agent section
(`docs/preregistration_llm_agent.md`, arm B in `evaluation/llm_agent_eval.py`).

### Testing

Tests live in `tests/`. They use small hand-made tables, need no network and
no API key, and run in seconds:

```bash
pytest -q                                              # everything
pytest -q tests/test_replay.py                         # one file
pytest -q -k future                                    # tests whose name matches (the leakage tests)
```

The GitHub Actions workflow was removed in the layout refactor. Restoring
`.github/workflows/tests.yml` to run `pytest -q` will bring back the PR check.
Add a test with every package: a planted failure that the code must catch is
worth more than a test that only runs the happy path.
