# Quick wins (branch `chore/quick-wins`)

No LLM API calls were made for any of this ($0 spend); every LLM number below is replayed from `runs/llm_cache/`.
Full test suite on the final branch: 417 passed (404 after the merge + 1 banned-word test + 12 shift-baseline /
pre-filter / default tests in `tests/test_shift_baseline.py`).

## 1. DeepSeek migration merged

`chore/deepseek-v4-models` merged into `chore/quick-wins` (fast-forward from `models/m1-baselines`, no conflicts).

| Check | Result |
| --- | --- |
| Full test suite after the merge | 404 passed |
| Published LLM + sceptic run (`test-reasoner-full/tool_sceptic`) replayed from the cache with every HTTP request refused (`python -m evaluation.llm_replay_check`) | 0 HTTP requests, 0 cache misses, 771 / 771 decision points, fills identical (4 / 4) |
| Published Dec–Jan run (`prev-tool_sceptic/tool_sceptic`, walk-forward M4), replayed on the final code of this branch | 0 HTTP requests, 0 cache misses, fills identical (4 / 4); 631 decision points vs 630 in the published trace (the extra point made no uncached LLM request) |

## 2. "locked" banned-word gap

`BANNED` (defined once in `agents/graph.py`, imported by `tool_agent`, `coach`, `scorer`) is now
`\block(?:s|ed)?\b|\block[- ]?in\b|guarantee|can'?t lose|sure thing|risk[- ]free`. Tests: lock, locks, LOCKED,
lock-in, "lock in" and lockin are caught; block, blocked, blocks, padlock, clock, unlocked, lockdown, Locke and
Lockett are not.

| Scan (`python -m evaluation.quick_wins banned`) | Newly matched |
| --- | --- |
| Proposal rationales and sceptic reasons in every `runs/**/trace.jsonl` | 0 |
| All 12,848 cached LLM replies (`rationale` / `reason` fields) | 0 |
| Published decisions that would change | 0 |

No cached reply contains "lock" in any form, so the published LLM runs are unchanged.

## 3. M4 news-shift double-count

**Bug.** The anchored agent prices `p = anchor + M4(out now) − M4(out = [])`, where the anchor is the Kalshi mid 24 h
before tip. Any absence the market already knew at the anchor is counted again.

**Data constraint.** Every news row in the frozen feed is an ESPN inactive list stamped exactly 30 min before tip
(3,371 rows, all `out`), so a game's own news is never public at the anchor: a literal "this game's news at or
before tip − 24 h" baseline is always empty and would change nothing. What is public at the anchor is each team's
previous inactive list; 41% of all listed absences (1,372 / 3,371) were already on the team's previous list.

**Fix.** `MarketAgent(shift_baseline="anchor")` (opt-in; `--shift-baseline` on `agents.graph`) uses
`anchor_out(view, game)`: for each team, the latest status per player (`OUT_STATUSES`) from that team's most
recent game tipping at or before the anchor, using only news published at or before the anchor. The shift is
then `M4(out now) − M4(out at anchor)`, i.e. only news since the anchor moves the price. Until tonight's list is published (and for games with no list in the
feed) nothing new is known and the shift is 0; without that rule (first run below) every player out last game
looked like a return. `shift_baseline="empty"`
reproduces the published runs. Gate back-test copies inherit the flag.

**Tool agent.** `AsOfTools.m4_win_prob` had the same empty "before"; it now takes the same flag but defaults to
`"empty"` (and `llm_agent_eval --shift-baseline` defaults to `empty`), because changing the tool output changes the
prompts and so the cache keys: the published LLM runs would no longer replay for free. The LLM arms were **not**
re-run with `"anchor"` (that would cost money). `agents/coach.py` (demo coach) still uses the empty baseline.

**Re-run: deterministic anchor agent, no learning, test 1 Feb – 12 Apr 2026** (fixed 4¢ edge, flat $20; same agent
as `adaptive_edge` fixed4 / `llm_agent_eval` setup A, run day-chunked through `llm_agent_eval --setups anchor`;
day-clustered bootstrap, 2,000 reps, seed 7606).

| Shift baseline | Trades | Staked (incl. fees) | P&L after fees [95% CI] | CLV $ [95% CI] | Mean CLV / contract [95% CI] |
| --- | --- | --- | --- | --- | --- |
| `empty` (published) | 129 | $2,543 ($2,645) | −$247 [−718, +265] | −$35 [−60, −12] | −0.35¢ [−0.65, +0.00] |
| `anchor` (fixed) | 121 | $2,386 ($2,488) | −$438 [−956, +82] | −$36 [−64, −10] | −0.26¢ [−0.59, +0.14] |
| `anchor` without the "no list yet, no shift" rule (discarded first run) | 164 | $3,239 ($3,380) | −$846 [−1,432, −244] | −$56 [−90, −23] | −0.34¢ [−0.66, +0.03] |

`empty` reproduces the published run exactly (same 129 fills as `runs/adaptive_edge/fixed4`). The fixed run trades
99 of the same games, drops 30 and adds 22.

| Paired, `anchor` − `empty` (same game-days) | Difference [95% CI] | p (one-sided, anchor > empty) |
| --- | --- | --- |
| P&L | −$191 [−446, +50] | 0.93 |
| CLV $ | −$1 [−15, +11] | 0.58 |
| Mean CLV / contract | +0.09¢ [−0.06, +0.27] | 0.16 |

**Brier of the forecast p = clip(anchor + shift) on the 501 test games** (`evaluation/results/m4_games.csv` + 24 h
anchor, as in `forecast_blend.md`; M4 "before" recomputed, matching the published values to 1e-16):

| Shift baseline | Brier | Difference vs `empty` [95% CI] |
| --- | --- | --- |
| `empty` (published) | 0.1663 | – |
| `anchor` | 0.1690 | +0.0027 [+0.0006, +0.0048] (worse) |

203 of the 501 games had an absence already known at the anchor; the mean |shift| falls from 1.97 to 1.80 points.

**Reading.** Removing the double count does not help. CLV is unchanged (−$36 vs −$35) and P&L is lower but within
noise, while the forecast's Brier is *significantly worse*. So the 24 h anchor does not fully price an absence from
the previous game: whether the player sits again is still uncertain a day out, and counting the full M4 effect again
was closer to the truth than counting none. A partial baseline (e.g. half the anchor-time effect, or the probability
the player is still out) might sit in between; it is not tried here because it would be tuned on the test period.
The evidence favours `empty`. Split-gate learning was not
re-run with the fix (a single sequential replay with gate back-tests did not finish in the time available).

**Default reverted to `empty`.** Anchor baseline tested (121 trades, −$438, Brier 0.1690 vs 0.1663): worse, not default.
`anchor` stays available as an opt-in (`MarketAgent(shift_baseline="anchor")`, `--shift-baseline anchor`).
Check: setup A re-run with `MarketAgent()` built without `shift_baseline` reproduces the published anchor agent
exactly (129 fills identical on every column to `runs/quick_wins/shift_fixed4_empty`; P&L −$247, CLV −$35).

## 4. Deterministic pre-filter before the LLM

`ToolAgent(prefilter_edge=x)` / `PlainLLMAgent(...)` / `llm_agent_eval --prefilter-edge x` (default off): before
any LLM call, compute the deterministic anchor-agent edge (`AsOfTools.anchor_edge`: anchor + M4 shift on the news
so far, best side, after fees, empty baseline as published) and skip the decision point without calling the
LLM if it is below `x`. Not callable by the LLM (not in `TOOL_SPECS`).

Offline evaluation on the two published DeepSeek-reasoner + sceptic runs (test 1 Feb – 12 Apr: 771 decision
points; Dec – Jan: 630), `python -m evaluation.quick_wins prefilter`; 1,397 points reached the LLM, using 5,187
LLM calls and $9.29 (estimated, at list price); 182 proposals reviewed by the sceptic, 8 approved and traded.

| Threshold | Points skipped (test / Dec–Jan) | LLM calls saved | $ saved | Reviewed proposals lost | Approved trades lost | CLV $ of lost trades (all 8: +$7.38) | P&L of lost trades (all 8: +$39) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1¢ | 682 / 1,397 (49%): 361 / 321 | 2,469 / 5,187 (48%) | $3.97 / $9.29 | 34 / 182 | 3 / 8 | +$3.20 | −$62 |
| 2¢ | 875 (63%): 459 / 416 | 3,159 (61%) | $5.17 | 44 | 6 / 8 | +$8.15 | +$80 |
| 3¢ | 1,020 (73%): 553 / 467 | 3,696 (71%) | $6.22 | 58 | 6 / 8 | +$8.15 | +$80 |

Deterministic edge of the 8 approved trades: +6.9¢, +5.1¢, +2.0¢, +1.2¢, +1.8¢, +0.9¢, −0.5¢, −1.1¢. Only the two
the deterministic agent would also have taken (≥ 4¢) survive a 2¢ filter, and those two have CLV −$0.77 together;
the other six carry all of the sceptic agent's positive CLV (+$8.15) and its one big winner (+$121).

**Reading.** The pre-filter saves about half the LLM spend at 1¢, but the LLM + sceptic agent's value (small as it
is: 8 trades, CLV +$7.38) comes from trades where the deterministic edge is small or negative, i.e. where it
disagrees with anchor + M4. Filtering on the deterministic edge removes exactly those. Recommendation: keep it off;
at most 1¢ if cost matters more than the (statistically weak) CLV of 3 trades. Caveat: 8 trades is far too few to
estimate the lost CLV precisely.
