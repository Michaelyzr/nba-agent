# Pre-registration: H (DSL rules) and I (episodic memory)

Group 12. Committed **before** any test-period evaluation of either component.
Date: 7 Oct 2026. Presentation: Fri 9 Oct.

## Windows (frozen; not tuned on)

| Split | Dates | Role |
| --- | --- | --- |
| Development | 1 Nov 2025 – 31 Jan 2026 | Prompt / search freeze, warm-start notebook and memory |
| Test (main) | 1 Feb 2026 – 12 Apr 2026 | Report numbers here |
| Play-off holdout | 13 Apr 2026 – 14 Jun 2026 | Scored once previously; **do not tune**, report only if time |

Same replay as the rest of the project: `Replay` with the `AsOf` view, fills at the ask
plus the Kalshi fee, stake $20, market-anchored M4 forecaster (`Forecaster.load()`), offline
unless `GEMINI_API_KEY` is set. Statistics: `evaluation/stats.py` day-clustered bootstrap,
2,000 replicates, seed 7606.

## Hypotheses

**H0 (shared).** Neither component beats the current offline template-learning agent
(`MarketAgent` with `learn=True`) on **CLV dollars** over the test window; the 95% CI of
(component − template) includes zero or is below it.

**H1 (DSL rules).** An agent whose reviewer proposes free-form rules in a safe DSL, which
are parsed, validated, and still gated, has higher CLV dollars on the test window than
(a) the same agent with no learning and (b) the template reviewer. Secondary: its parse-valid
rate is high and its gate acceptance rate is not higher than the template reviewer's.

**H2 (episodic memory).** An agent that retrieves the k nearest settled episodes at decision
time and skips a trade when their mean CLV is clearly negative has higher CLV dollars on the
test window than (a) no learning and (b) the template reviewer. Secondary: vetoes that would
have been filled have mean CLV ≤ 0 (the veto was directionally right).

A result that fails H1 or H2 is reported as a null. That is an acceptable outcome.

## Metrics (pre-declared)

For every setup, over the same game-days:

- CLV dollars (primary evidence gate vs never-trade and vs template learning)
- mean CLV per trade
- P&L after fees
- number of trades
- day-clustered bootstrap 95% CI and one-sided p (H1: value > 0 / a > b)

DSL-only process metrics:

- proposals written, parse-valid rate, gate pass rate (accepted / gated), rules active at end

Memory-only process metrics:

- recalls with enough history, vetoes, mean realised CLV of vetoed candidates (shadow CLV)

## Component specs (frozen)

### H — safe DSL

- Grammar: conjunction of at most 3 comparisons over
  `{side_price, gap, market_move, model_shift, hours_to_tip, news_age_minutes}` with `<=` /
  `>=` and bounded constants, plus `market_kind == game`; actions `skip`, `min_edge`,
  `stake_scale` with bounded constants. Hand-written parser; never `eval`.
- Compile target: existing notebook `when` / `do` JSON; `notebook.validate` runs again.
- Proposals per day: K = 3. Invalid text is logged with a reason and never reaches the gate.
- Proposer: Gemini via `llm.ask` when `GEMINI_API_KEY` is set (responses cached under
  `runs/llm_cache/`, key = sha256 of model + prompt + temperature 0); otherwise a
  deterministic grid search over the same language (`stub_proposer`).
- Gate: use the de-overlapped total-CLV-dollars gate if the safety agent has landed it in
  `MarketAgent.gate` by evaluation time; otherwise the existing mean-CLV gate. Which one
  was used is recorded in the results file. No gate code is changed by this work.

### I — episodic memory

- Features: `(side_price, gap, market_move, model_shift, hours_to_tip, log1p(news_age))`,
  z-scored on the episodes visible at `now`. Similarity: Euclidean. k = 20.
- Store an episode only once `final_at <= now`. Recall only sees `settled_at <= now`.
- Deterministic adjustment (not LLM context): skip if at least 40 settled episodes exist
  and the k neighbours' mean CLV ≤ −0.01. Memory can only remove a trade.
- Shadow episodes: a vetoed candidate is stored with the CLV it would have had once the
  game settles, so later recalls see it.

## Setups on the test window

1. `no_learning` — `MarketAgent(learn=False)`, empty notebook
2. `template` — `MarketAgent(learn=True)`, REVIEW_TEMPLATES reviewer (current agent)
3. `dsl` — `DSLReviewAgent` (H)
4. `memory` — `MemoryAgent(learn=False)` (I alone)
5. `memory_template` — `MemoryAgent(learn=True)` (I + template reviewer)

Warm-start: for learning setups, run development first and carry the notebook (and the
memory bank for I) into the test window. `no_learning` and cold `memory` start empty.

## What we will not do

- Tune K, bounds, SKIP_CLV or the stub grid on the test window.
- Score the play-off holdout more than once, or use it to choose a setup.
- Edit `agents/graph.py`, `agents/notebook.py` or `replay.py` except for a listed tiny
  additive hook if a plug-in path is missing.
- Print or commit API keys, `.env`, or `runs/llm_cache/`.
