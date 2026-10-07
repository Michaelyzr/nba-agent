# Pre-registration: LLM tool agent with priced-in sceptic

**Date:** 2026-10-07  
**Branch:** `models/m1-baselines`  
**Review:** `docs/agentic_review.md`, option A (tool-using LLM agent) and F lite (priced-in sceptic)  
**Code:** `agents/tools.py`, `agents/llm_client.py`, `agents/tool_agent.py`, `evaluation/llm_agent_eval.py`

This file is committed **before** any test-period or holdout Gemini run. Prompts are frozen here and in
`agents/tool_agent.py` (`ANALYST_SYSTEM`, `SCEPTIC_SYSTEM`, `PLAIN_SYSTEM`). Development days
(November–January) may be used only to confirm the protocol and the cache, not to retune the prompts
for the test numbers.

---

## Question

Does a tool-using LLM that *proposes* trades (not only vetoes them), under the same as-of tools,
fee formula, min-edge rule and risk caps as the deterministic market-anchor agent, beat that agent
or never-trade on closing-line value? Does a second "priced-in sceptic" call improve the result?

## Setups (arms)

| Code | Setup | Learning |
| --- | --- | --- |
| A | Deterministic `MarketAgent`: market mid 24 h before tip + M4 news shift, offline rules | Off |
| B | Plain LLM: one Gemini call with quote, anchor, news and M4 before/after as text (README "ChatGPT baseline"); same trading rule | Off |
| C | Tool agent: Gemini chooses as-of tools, proposes `{buy\|pass, estimate_p, citations}`; code validates and applies the gap rule | Off |
| D | C + priced-in sceptic (second LLM call; can only reject) | Off |
| C-anon | Optional: C with team labels `TEAM_A`/`TEAM_B` and no game date (memorisation check) | Off |
| never | Never trade | – |

Identical for every arm: Replay decision times, stake $20, caps $50 / $100 / $300, fills at the
ask (or 1 − bid) plus the Kalshi fee, volume cap, one position per game, default M4 (trained on
games before 1 Feb 2026). M6 is exposed as a tool if the pickle exists; it is not required to
clear the gap rule. Default: no notebook learning, so the LLM effect is isolated.

## Windows

| Window | Dates | Role |
| --- | --- | --- |
| Development | 1 Nov 2025 – 31 Jan 2026 | Prompt freeze and protocol checks only. A short January subsample may be run to confirm the cache and the JSON schema. |
| **Test (main)** | **1 Feb – 12 Apr 2026** | Primary comparison. Same 501 Kalshi games / ~771 decision points as the existing test runs. |
| Holdout | 13 Apr – 14 Jun 2026 | Scored once if API budget allows. Marked "seen once" in the report (the play-off holdout has already been scored for the deterministic agent). |

If Gemini quota cannot cover ~771 decisions × several calls, we subsample decision times with a
**fixed seed (7606)** and the **same** subsample for every setup, and we say so in the report.
Subsampling is not re-tuned on the test outcomes.

## Hypotheses (pre-registered)

CLV is per contract against the mid at tip. Orders fill at the ask, so a trader with no edge over
the close scores about −½ spread (~−0.005). Dollar CLV is CLV × contracts. P&L is after fees.
Statistics: day-clustered bootstrap, 2,000 replicates, seed 7606 (`evaluation/stats.py`). Paired
comparisons use the same resampled game-days. One-sided p-values are for "first > second".

| Id | Hypothesis | Primary metric | Decision rule |
| --- | --- | --- | --- |
| **H1** | The tool agent (C or D, whichever is the main LLM arm) does **not** beat A on CLV dollars. | paired CLV $ of LLM arm − A | Fail to reject if 95% CI of the difference includes 0 or the estimate is ≤ 0. Report both C and D. |
| **H2** | The sceptic (D) reduces the number of trades relative to C without lowering mean CLV per trade. | trades(D) < trades(C); paired mean CLV of D − C | "Supports" if trades fall and the mean-CLV CI includes 0 or is above 0. |
| **H3** | No LLM setup (B, C, D) beats never-trade on P&L or CLV dollars. | paired P&L and CLV $ of each LLM arm − never | "Supports" if every LLM arm's CI for the difference is not strictly above 0. |
| **H4** (secondary) | Invalid / hallucinated proposals are caught by code validation (cited-number mismatch, ungrounded estimate, banned wording). | invalid rate; share of invalid that are `number_mismatch` or `ungrounded_estimate` | Descriptive. |
| **H5** (secondary) | Anonymising teams and hiding the date (C-anon) does not change the qualitative conclusion of H1/H3. | same metrics on C vs C-anon | Run only if budget allows. |
| **H6** (secondary) | A 20-question memorisation probe ("who won X at Y on date D?" on randomly drawn test games, with an explicit "unknown" option) scores near the home-team / favourite baseline, not near 100%. | accuracy among answered and among all questions | Descriptive leakage check. |

We do **not** claim an edge. A null or a loss for the LLM agent is a reportable finding (free agency
is harmful or unused in an efficient market). The evidence gate is the same as for every other
component: beat A and never-trade on CLV dollars with a CI above zero, or report a null.

## Metrics (reported for every arm)

- Trades (fills).
- Mean CLV per trade with day-clustered 95% CI; one-sided p for CLV > 0.
- CLV dollars and P&L after fees, each with CI.
- Paired differences vs A and vs never-trade (mean CLV, CLV $, P&L).
- Agent diagnostics (LLM arms): tool calls per decision, buy proposals, invalid-output rate by
  reason, sceptic reject rate, mean CLV of vetoed vs kept trades (counterfactual tip mid − would-be
  fill price for vetoes), share of fills on games A did not trade, LLM call count, cache hits,
  tokens in / out, estimated USD cost, latency per call.

## Protocol details (frozen)

- Model: `gemini-2.5-flash` (override only via `GEMINI_MODEL`), temperature 0, JSON response mime
  type, thinking budget 0.
- Cache: `runs/llm_cache/`, keyed by SHA-256 of (model, temperature, role, system, user prompt).
  Errors are never cached. Re-runs are deterministic and free.
- Tool budget: at most 8 tool calls and 4 turns per decision, then a forced final.
- Validation: schema; citations must name an executed call and match its output; rationale numbers
  must appear in tool outputs; `estimate_p` must lie within 0.03 of a tool-derived probability
  (market mid, anchor, anchor + M4 shift, M4 after, or M6 tip estimate); banned wording; news ids
  in the rationale must be public by `now`. Fail → pass with a logged reason.
- Gap rule: `estimate_p − price − fee > 0.04` (same default as `MarketAgent`), then `pretrade_risk`
  and Replay caps.
- Sceptic: approve or reject only; invalid sceptic output fails closed (reject).
- Leakage: tools wrap `AsOf`; prompts use relative times. Team names and the game date remain in
  the default prompts, so pretraining-knowledge leakage is possible; H5 and H6 address that.

## What we will not do after seeing the test numbers

- Change the prompts, the grounding band, the min edge or the tool list to improve the test table.
- Drop or reweight arms post hoc.
- Treat the original test window as a clean holdout (it is not; the anchor was designed after an
  earlier look at it). The walk-forward and the play-off holdout remain the honest long-horizon
  checks for the deterministic agent; this pre-registration covers the LLM comparison on the same
  windows the review specifies.

## How to run (after this commit)

```bash
# development protocol check (heuristic stub, no API key)
python -m evaluation.llm_agent_eval --window test --backend heuristic --name dryrun --stem llm_agent_dryrun

# test window (requires GEMINI_API_KEY in .env)
python -m evaluation.llm_agent_eval --window test --backend gemini --setups anchor,plain,tool,tool_sceptic

# holdout once, if budget remains
python -m evaluation.llm_agent_eval --window holdout --backend gemini --setups anchor,plain,tool,tool_sceptic

# memorisation probe
python -m evaluation.llm_agent_eval --probe 20 --backend gemini

# re-score without replaying
python -m evaluation.llm_agent_eval --report-only --report test,holdout
```

Outputs: `evaluation/results/llm_agent.md`, `llm_agent.csv`, `llm_agent.png`; run artefacts under
`runs/llm_agent/<window>/` (git-ignored).
