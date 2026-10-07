# Agentic framework review (Group 12, 7 Oct 2026)

Scope: `agents/graph.py`, `agents/notebook.py`, `agents/coach.py`, `agents/league.py`,
`agents/pregame.py`, `agents/inplay.py`, `replay.py`, `llm.py`, `forecast/` (M1–M6) and
`evaluation/results/*.md`. This is a design review only; no code was changed.

The rubric is the Track 2 table in `README.md` §8: Problem 15, Tool design 15, Deep learning 20,
Implementation and demo 25, Evaluation 15, Report and Q&A 10. It also says the work "must go
beyond a basic model or API call".

---

## 1. Verdict

**The evaluation is strong, but the agent is simple.** A demanding examiner will see a
deterministic pipeline with an LLM layer on top. The rigour of the replay and the honest null
results will impress them. Then they will ask what the agent does that a 50-line script could not.
At the moment, the honest answer is "very little".

How an examiner reading `agents/graph.py` will see it:

1. **Fixed graph.** Edges are hard-wired: trigger → investigate → forecast → analyse →
   propose/no_action → checks → risk → confirm. The only branches are a threshold
   (`gap > min_edge`), a check failure and an empty order list. The agent never chooses which tool
   to call, what data to fetch or when to stop. These are the core features of "agentic framework
   design".
2. **The LLM can only veto.** `_analyse_llm` can only turn `act` from true to false. `_investigate_llm`
   must reproduce the same player list as the structured statuses, or the code raises an error and
   falls back. The safety design is principled, but the LLM adds almost nothing to the decisions.
3. **No reported number uses the LLM.** Every results table says "Offline rules; no LLM".
   `evaluation/ablations.py --llm` exists but has never been run for a result. The LLM agent is
   therefore unevaluated.
4. **The retry is a no-op offline.** After a failed check, `investigate` is re-run with the same
   inputs and gives the same output. Only planted faults, which fire on try 0 only, make the retry
   succeed. An examiner will spot that this "self-correction" is staged.
5. **"Learning" means picking one of 10 hand-written templates.** `REVIEW_TEMPLATES` lists every
   rule the offline reviewer can propose. Every template either skips a slice of trades or raises
   the minimum edge. The results confirm what this implies: learning only removes trades.
6. **The gate is biased toward accepting abstention.** Two separate problems:
   - **Selection and test data overlap.** `_review_offline` picks the worst slice from the last
     21 days (`REVIEW_WINDOW`). `gate` then back-tests on the last 14 traded days (`GATE_DAYS`),
     which lie inside those 21 days. The rule is tested partly on the data that selected it. This is
     not look-ahead leakage, but it is in-sample selection bias.
   - **The metric rewards dropping trades.** The gate uses mean CLV per trade, with no penalty for
     trading less. Removing a slice that happened to have negative CLV raises the mean by
     construction. With 37 proposals and 5 accepted, we cannot currently show the acceptance rate
     is above chance.
7. **Deep learning is not on the decision path.** Trades are driven only by M4, a logistic
   regression (`forecast/win.py`). The neural models do not affect trades: the M2 GRU and M3 are
   not used, and M6 ImpactNet makes 0 trades. For a deep-learning course, that puts the DL
   criterion (20 points) at risk.
8. **Some safety claims are not implemented.** The README lists a "kill switch" and "cool-downs".
   `scorer.py` only counts days with losses over $100 after the run; nothing stops trading. Retail
   confirmation in the replay is `auto_confirm`, which returns every order. The policy tests (9 of
   9 blocked) check the caps, not these claims.
9. **Promised baselines are missing.** README §6 promises a ChatGPT baseline and reviewer blame
   accuracy on 30 hand-labelled trades (E2). Neither exists in the code (`grep` finds no
   `chatgpt`, no blame-accuracy code and no `evaluation/labels`).
10. **The three agents are separate systems.** Pregame (fair odds from news), in-play and the
    market trader are separate LangGraphs with separate state. There is no orchestrator, and the
    pregame news factors (questionable or doubtful weights) never reach the trader.

**What is above average and should be defended hard:**
- **As-of replay:** an `AsOf` view, fills at the ask plus the Kalshi fee, volume caps, and
  leakage tests.
- **Statistics:** day-clustered bootstrap, a walk-forward with M4 retrained before each month,
  and a play-off holdout scored once.
- **Honesty:** a research-process disclosure.
- **A clear positive finding:** the market anchor (+$2,450, p = 0.002).
- **A rigorous null for M6:** quantile GRU with pinball loss and four baselines, which
  establishes a martingale result.
- **Consumer-fairness evidence:** about 1.9¢ all-in cost per contract, and 91% of the price move
  happens before the inactive list is public.

Many groups will have nicer agents and much weaker evaluation. Our risk is the reverse: strong
evaluation of a weak agent, and the course is named "Agentic Framework Design".

---

## 2. Map to grading criteria

The low and high ranges are estimates of where an examiner might land. "Now" assumes no further
changes; "with plan" assumes the 72-hour plan in §4.

| Criterion (pts) | Now | With plan | Why it scores high | Why it scores low |
| --- | --- | --- | --- | --- |
| Problem and user need (15) | 10–13 | 11–14 | Clear thesis backed by data (costs, information speed); a consumer-education product | Pivot from "trading agent" to "education" is late; the Coach has no user evidence |
| Tool design and usability (15) | 9–12 | 11–13 | Coach and League are coherent; per-channel briefs; live Polymarket page | Four separate apps or tabs; pregame, trader and coach are not one workflow |
| Deep learning (20) | 9–13 | 11–15 | GRU distribution model, ImpactNet with quantile loss and an honest null | Trades run on logistic regression; DL never affects a decision |
| Implementation and demo (25) | 16–21 | 18–22 | End to end on real Kalshi data; LangGraph; tests; planted failures | Agent is a fixed pipeline; LLM path never demonstrated with numbers |
| Evaluation and limitations (15) | 12–14 | 13–15 | Walk-forward, holdout, clustered CIs, disclosure | Gate double-dips; missing ChatGPT baseline; "kill switch" only measured |
| Report, presentation, Q&A (10) | 6–8 | 7–9 | Strong narrative if framed as "the market is the grader" | Negative headline P&L can read as "nothing works" if badly framed |
| **Total** | **62–81** | **71–88** | | |

The biggest lever per hour is **Implementation (agentic depth)**. Next come **Evaluation**,
cheaply, by fixing the gate and the safety gaps, and **DL**, by putting a learned component on
the decision path or framing M6 as the DL experiment it is.

---

## 3. Improvement options

Every option is evaluated the same way. This is the leakage-safe protocol in this section:

- **Same replay.** Use `Replay` with the `AsOf` view, fills at the ask plus fee, and the same
  caps.
- **Same windows.** Use the walk-forward season (1 Nov – 12 Apr, M4 retrained monthly) as the
  main test. Also report the original test window, marked "not clean". The play-off holdout has
  been scored once already, so call it "seen once" and do not tune on it.
- **Same statistics.** Use `evaluation/stats.py` paired day-clustered bootstrap, 2,000 replicates
  and seed 7606. Report against **never trade** and against the current anchor agent:
  - CLV dollars;
  - mean CLV per trade **plus half the spread**, i.e. edge over the mid, where zero means no edge;
  - P&L after fees;
  - number of trades.
- **Pre-registration.** Write the hypotheses, thresholds and prompts into a dated file and commit
  it before the run.
- **The new component must pass the same evidence gate as everything else.** If it does not beat
  the current agent on CLV dollars with a CI above zero, report it as a null result.

### Ranking (best marks per hour first)

| Rank | Option | Effort (h) | Fri/Sat feasible? |
| --- | --- | --- | --- |
| 1 | A. Tool-using LLM agent vs the deterministic agent, with a sceptic step | 10–14 | Yes (core build) |
| 2 | B. Safety and gate audit: enforced kill switch, invariants, placebo gate, de-overlapped gate | 5–7 | Yes (cheap, must do) |
| 3 | C. Agentic Coach tutor with simulated-user evaluation (plus a 5-person pilot) | 8–12 | Partly (simulated users only) |
| 4 | D. Fractional Kelly with calibrated probabilities and abstention | 3–5 | Yes |
| 5 | E. Orchestrating pregame → trader → coach into one multi-agent system | 6–10 | Thin version only |
| 6 | F. Full multi-agent deliberation (analyst, priced-in sceptic, risk manager) | 8–12 | Lite version via A |
| 7 | G. Learned trade/pass and sizing policy (full-information offline learning) | 10–16 | Future work |
| 8 | H. LLM free-form rules compiled to a safe DSL, still gated | 6–10 | Future work |
| 9 | I. Episodic retrieval memory | 6–10 | Future work |

---

### A. Tool-using LLM agent vs the deterministic agent (rank 1)

**What.** A ReAct- or function-calling agent that chooses which as-of tools to call, builds its
own case and may *propose* a trade, not only veto one. Code still computes every number, and
checks and risk still have the final word. A second short LLM turn acts as a "priced-in sceptic"
and argues against each proposed trade (this is option F, lite).

**Why it raises marks.** It is the single change that turns "a pipeline with an LLM layer" into
agentic framework design: tool selection, planning, a stopping decision and self-critique. It
also produces the missing number, *LLM agent vs deterministic agent vs never trade*, under the
same rules. Either outcome is reportable:
- If the agent trades more and loses the spread, that is evidence that free LLM agency is harmful
  in efficient markets. This is a strong, on-thesis finding.
- If it abstains as well as the rules do, the deterministic agent is enough at a fraction of the
  cost.

**Design in our code.**
- **New file `agents/tool_agent.py`.** Add a `ToolAgent` class with the same `policy(view, game,
  now)` interface, so `Replay` and `evaluation/ablations.py` work unchanged.
- **Tools.** Thin wrappers that close over the `AsOf` view, so leakage is impossible by
  construction:
  - `get_news(game_id)`, `get_quote(ticker)`, `get_price_path(ticker, hours)` and
    `get_anchor(ticker)`, which reuse `MarketAgent._anchor_mid`;
  - `forecast_win(out_player_ids)`, which calls `forecast/api.py` (M4);
  - `team_form(team)`;
  - `lookup_rules(situation)`, which calls `Notebook.matching`;
  - `compute_gap(ticker, side, p_source_id)`, where code applies the fee formula;
  - `propose_order(ticker, side, p_source_id, reason)` and `pass(reason)`.
- **Number provenance.** `propose_order` takes the *ID* of an earlier `forecast_win` or
  `compute_gap` result, never a free number. `checks` already rejects a `p_model` that is not the
  model's, so extend it to match on provenance IDs.
- **Budget.** At most 8 tool calls per decision, then a forced `pass`. Log every call in `trace`,
  as the current trace does.
- **Determinism.** Use temperature 0 and cache responses keyed by a hash of the prompt and tool
  results (for example in `runs/<name>/llm_cache.jsonl`), so reruns and the bootstrap reuse
  identical decisions. `llm.py` needs a `tools=` path; Gemini supports function declarations.
- **Sceptic.** For each proposed order, one call gets the order, the price path since the anchor
  and the news timing, and returns `{keep: bool, why}`. Like the current analyst, it can only
  drop an order.

**Evaluation.**
- **Arms:** (1) deterministic anchor agent with no learning; (2) tool agent; (3) tool agent plus
  sceptic; (4) never trade. Use identical caps.
- **Windows:** the original test window (501 games, about 1,300 decision points) as the main run
  if the API budget is tight; add walk-forward months if there is time.
- **Paired bootstrap** of CLV dollars and edge over the mid against arms 1 and 4.
- **Agent diagnostics:**
  - tool calls per decision;
  - percentage of decisions where the agent *added* a trade the rules would not take;
  - number of checks failures caught (hallucinated numbers);
  - sceptic veto precision, i.e. mean CLV of vetoed trades versus kept trades;
  - cost and latency per decision.
- **Leakage:**
  - The tools only expose `AsOf`, and the existing future-row test applies.
  - Gemini 2.5 Flash's training cutoff is before the 2025-26 season. Still run a 20-question
    probe ("who won X at Y on date D?") and report the accuracy. Chance means no recall.
  - Optionally anonymise team names in the prompts as a robustness arm.

**Effort.** 10–14 hours for two people: about 6 to build, 2 for the cache and checks, 2–4 for the
runs, and 2 for the statistics and slide.

**Risk.**
- **Medium.** API rate limits and cost could bite. At about 1,300 decisions × 4 calls, Flash is
  cheap, but the run has to be sequential per day.
- **Prompt iteration on the test window is itself tuning.** Freeze the prompts on November–January
  first.

**Feasible by Fri/Sat:** yes, as the main build. Cut-down version if needed: the tool agent on the
test window only, without the sceptic.

---

### B. Safety and gate audit (rank 2)

**What.** Four small fixes that close the gaps between what we claim and what the code does:

1. **Enforced kill switch.** Stop trading for the rest of the day once realised losses exceed
   $100, then add an N-day cool-down after K tripped days.
2. **Formal invariants.** Property-based tests with Hypothesis over random orders and states:
   - no fill at or after tip-off;
   - stake, game and day caps hold;
   - the LLM cannot add an order (analyst and sceptic);
   - every cited news item is public at decision time;
   - rule `valid_from` is at or before the decision time.
3. **De-overlapped gate.**
   - The reviewer selects a slice on days [t−35, t−14). The gate tests it on [t−14, t).
   - The gate metric becomes **CLV dollars against the run without the rule**, or edge over the
     mid. This penalises trading less, so it no longer rewards dropping trades.
4. **Placebo gate.** Feed the gate 200 rules that skip a *random* slice of the same size, and
   compare their acceptance rate with the real reviewer's 5 of 37.

**Why it raises marks.**
- It turns two awkward Q&A moments ("is the kill switch real?" and "is the gate just overfitting?")
  into rigour points.
- A placebo test is exactly what a demanding examiner hopes to see.
- It is cheap, and it does not risk the demo.

**Design in our code.**
- **Kill switch.**
  - Add a `risk` callable in `replay.py`, or extend `basic_risk`, that reads a ctx field
    `realised_day`. That field may only use P&L of games whose `final_at <= now`.
  - **Leakage trap:** `_settle` computes P&L at fill time from the outcome. The kill switch must
    not read `day_fills` P&L directly.
  - Log `kill_switch` in `risk_result`, and make `scorer.py` count real trips.
- **Invariants:** `tests/test_invariants.py` with `hypothesis` (one new dev dependency).
- **Gate.**
  - Add a gate-window parameter in `MarketAgent.gate`.
  - In `_review_offline`, filter `recent` to `as_of < day − GATE_DAYS`.
  - Switch `ok` to compare `with_rule.clv·contracts` against `without.clv·contracts`.
- **Placebo:** `evaluation/gate_placebo.py` reuses `MarketAgent.backtest` with random-mask rules.
  The reviewer only knows the template slices, so implement placebo rules as a seeded hash of
  `decision_id` turned into a skip. That needs one new `when` field (`placebo_bucket`) in
  `WHEN_EQUAL`.

**Evaluation.**
- Rerun the full agent on the walk-forward with the fixed gate. Report rules proposed and
  accepted, and CLV dollars against no learning and against never trade (paired bootstrap).
- Report the placebo acceptance rate with a binomial CI.
- If the real reviewer's acceptance rate is not above the placebo rate, say so: "learning reduces
  exposure, but the gate cannot yet tell real lessons from noise".

**Effort.** 5–7 hours in total: kill switch 1.5, invariants 2, gate split and metric 1.5,
placebo 2. The walk-forward rerun takes about 20 minutes per setup.

**Risk.** Low. The results may weaken the "learning helps" claim. That is still better than an
examiner finding the overlap.

**Feasible:** yes, by Thursday night.

---

### C. Agentic Coach tutor with a simulated-user evaluation (rank 3)

**What.** Make the Coach an agent instead of a fixed set of lesson cards:
- It plans the next lesson from the user's mistake history in `coach.feedback`.
- It asks a check question before revealing the outcome.
- It answers free-text questions grounded only in the `snapshot` numbers, and the existing
  `BANNED` and number checks apply.

Evaluate it with LLM-simulated users of three personas:
- **chaser**, who backs moved prices;
- **long-shot lover**;
- **cautious**.

Each persona plays the same seeded League slates, with and without the Coach. Add a five-person
pilot of classmates with a pre and post quiz if time allows.

**Why it raises marks.** The deck says the product is Coach plus League, but it has *no evidence
of usefulness*, which is an explicit Evaluation criterion. A simulated-user study gives the
product a measured outcome:
- pass rate;
- mean CLV;
- share of long-shot and chasing trades;
- quiz score.

It also makes the Coach visibly agentic in the demo.

**Design in our code.**
- **`agents/coach_agent.py`.** A small LangGraph:
  `snapshot` → `diagnose` (from the user's trade history) → `choose_lesson` (LLM picks from the
  lesson IDs in `coach.lessons`) → `ask_check_question` → `reveal` (`paper_trade`) → `update_profile`.
- **Grounding.** The free-text answer may only quote numbers present in `snapshot`; reuse the
  `PCT` check from `graph.py`.
- **Simulator.** `evaluation/sim_users.py`:
  - each persona is a prompt with a biased policy;
  - after each Coach message the persona may update its behaviour;
  - the control arm sees only the raw price.
  - Slates come from `league.py`'s seeded slates on the test period.

**Evaluation.**
- **Paired design:** the same persona, slate and seed, with and without the Coach. Use 30 slates
  × 3 personas × 2 arms.
- **Outcomes:** change in pass rate, mean CLV per trade, and long-shot and chase share, with a
  bootstrap over slates.
- **Leakage:** slates only show as-of data, and `tests/test_league.py` already covers this.
- **Honest caveat:** simulated users show the Coach's *mechanism*, not human learning. Label it
  as such, and present any human pilot as anecdotal (n = 5).

**Effort.** 8–12 hours: Coach graph 4, simulator 3, runs and statistics 2–3, plus the pilot.

**Risk.**
- **Medium.** Simulated-user results can look circular, because the LLM is coached by an LLM.
  Mitigate by using a different model or temperature for the personas and measuring outcomes by
  *market* CLV, not LLM judgement.

**Feasible:** the simulated arm by Friday if a separate pair owns it. The human pilot is a stretch
for Saturday.

---

### D. Fractional Kelly with calibrated probabilities and abstention (rank 4)

**What.** Replace the flat $20 stake with fractional Kelly (¼):
f\* = (p − c) / (1 − c), where c = ask + fee.

Here p is the anchored probability *after* a calibration step. Abstain when the bootstrap
interval of p − c includes 0.

**Why it raises marks.** It is principled decision theory instead of a magic 4¢ threshold. It
also ties the DL models' calibration (reliability) to money: if p is badly calibrated, Kelly
overbets and the replay shows it.

**Design in our code.**
- **Sizing.** In `MarketAgent.analyse`, compute the stake from the gap instead of
  `self.stake * scale`. Keep the risk caps.
- **Calibration.** Fit isotonic or Platt regression of *the anchored p* against outcomes on
  November–January only (M4 is in-sample there, so note it). Uncertainty of p comes from
  bootstrapped M4 refits, about 50, giving a shift distribution.
- **Abstention:** trade only if the 10th percentile of (p − c) is above 0.

**Evaluation.**
- **Arms:** flat stake vs Kelly ¼ vs Kelly ¼ with abstention vs never trade.
- **Statistics:** paired bootstrap on P&L and CLV dollars, plus maximum drawdown.
- **Expected result:** close to never trade, because the gaps are tiny. That is fine, and it
  supports the thesis.

**Effort.** 3–5 hours.

**Risk:** low. **Feasible:** yes. It is a good third build if anyone is free.

---

### E. Orchestrating pregame → trader → coach into one multi-agent system (rank 5)

**What.** A supervisor LangGraph (`agents/night.py`) that runs for one game night. The flow is:
1. The pregame agent polls news and writes factors, including questionable or doubtful weights,
   and fair odds.
2. The trader consumes those factors as its investigation, instead of only out or doubtful
   statuses.
3. The Coach explains the trader's decision to the retail user.
4. After settlement, the reviewer and gate run.

The state is shared through typed messages.

**Why it raises marks.** It gives one coherent workflow from input to output (Implementation 25)
and a real multi-agent architecture. It also uses the teammates' merged work instead of leaving
it beside the system.

**Design.**
- **Adapter.** `PregameAgent` snapshots (`factors`, `p_home`) become the trader's `investigation`.
  Map lost share to `availability` in `forecast/win.py` `team_side`, which already supports it.
- **Coach.** It reads the trader's `trace` for its "what the agent did" panel, which partly exists
  already.

**Evaluation.**
- **Replay test:** the trader with pregame factors vs status-only, on the walk-forward.
  Questionable weights are uncalibrated, so pre-register them as-is.
- **Leakage:** pregame replay already rejects future items. Add a test that the supervisor never
  passes a snapshot later than `now`.

**Effort.** 6–10 hours for the full version; 3–4 hours for a thin "demo-night" orchestration with
no new evaluation.

**Risk.** Medium-high for demo stability two days before the presentation.

**Feasible:** the thin version by Friday. The full version with evaluation is future work.

---

### F. Multi-agent deliberation: analyst, priced-in sceptic, risk manager (rank 6)

**What.** Three LLM roles debate each candidate for up to two rounds:
- the **analyst** argues for the trade;
- the **sceptic** argues it is already priced in, citing the price path since the anchor and the
  news timing;
- the **risk manager** weighs bankroll, exposure and caps.

A code judge aggregates their votes, and only unanimity to trade passes. All roles can only drop
trades.

**Why it raises marks.** It is visibly "multi-agent", and our data supports the sceptic's job:
91% of the move happens before the list. The sceptic's claims can be *verified* against
`market_move`.

**Design.** Add nodes `debate_analyst`, `debate_sceptic` and `debate_risk` between `analyse` and
`propose` in `graph.py`, behind a `--deliberate` flag. Each role gets a role prompt, the same row
table and the price path.

**Evaluation.**
- Arms: no debate vs debate (both with the anchor and no learning) vs never trade.
- Metrics:
  - veto precision, i.e. mean CLV of vetoed vs passed trades;
  - CLV dollars against no debate;
  - agreement with the offline `market_move_max` rule;
  - cost.
- **Key caveat:** a veto-only debate can only reduce trades, so it will converge on never trade.
  Pre-register veto precision as the main metric, not P&L.

**Effort.** 8–12 hours for the full version. The lite version is the sceptic inside A, at about
2 extra hours.

**Risk:** medium. **Feasible:** the lite version only. The full three-role version is future work.

---

### G. Learned trade/pass and sizing policy (rank 7)

**What.** Replace the rules with a policy π(action | situation) over the actions pass, buy home
and buy away (optionally with a size), learned from data. The reward is CLV per dollar.

**Key insight.** This is *not* a bandit problem. We fill at small sizes against recorded quotes,
so the CLV of **every** action at **every** decision time is known from the price data. That gives
full-information counterfactual labels, so this is supervised learning of action values, not
off-policy RL with importance weights. There are about 3–4k decision points per season × 3
actions.

**Why it raises marks.** It puts deep learning on the decision path: an MLP over the situation
vector and M6's GRU price embedding. It is learned in place of hand-written rules. It also gives
a clean comparison: learned policy vs rules vs never trade.

**Design.**
- `forecast/policy.py` takes features from `forecast/impact.py` `build_row`, which already holds
  the leakage-tested price sequence, plus the situation dict from `analyse`.
- Targets are (close mid − ask − fee) for each side.
- The policy trades if the predicted lower quantile is above 0.
- Plug it in as `MarketAgent(policy_model=...)`.

**Evaluation.** Use the same splits as M6 (train before 15 Jan, validate on 15–31 Jan, test
February to 12 April, holdout seen once). Compare against the template rules, the anchor agent
and never trade.

**Honest expectation.** The target is M6's target (move to close) minus half the spread. M6
already showed that this cannot be predicted beyond zero, so the learned policy should learn to
always pass. That is a valid result, but it adds little beyond M6.

**Effort:** 10–16 hours. **Risk:** high for the effort. **Feasible:** future work. Mention it as the
principled generalisation of the gate.

---

### H. LLM-proposed free-form rules compiled to a safe DSL, still gated (rank 8)

**What.** Instead of 10 templates, the reviewer LLM writes a rule in a small grammar. For example:
`when side_price <= 0.3 and news_age_minutes > 30 do skip`, or arithmetic over situation fields.
A parser compiles the rule to the existing `when`/`do` JSON, rejecting anything else. The gate
still decides whether it is kept.

**Why.** "Learning" then means *search over a hypothesis space*, not picking a template, and the
safety story stays intact: the parser, `validate` and the gate.

**Design.** A Lark or hand-written parser in `agents/rule_dsl.py`; conjunctions only, with fields
from `WHEN_FIELDS` and actions from `ACTIONS`. Extend `_review_llm` to output DSL text, and limit
it to three proposals per day.

**Evaluation.** Use only the de-overlapped gate from B. Otherwise free-form search overfits even
faster. Compare acceptance rates and walk-forward CLV dollars against templates and the placebo
rate.

**Effort:** 6–10 hours. **Risk:** more search means more false discoveries unless B ships first.
**Feasible:** future work.

---

### I. Episodic retrieval memory (rank 9)

**What.** Store each settled decision's situation vector, trace and outcome. At decision time,
retrieve the k most similar past episodes (cosine over situation features, or an embedding of the
news text) and show them to the analyst or sceptic as precedents.

**Why.** It is a recognisable agentic-memory pattern, and a natural complement to the rule
notebook: memory gives soft precedents, rules give hard, gated constraints.

**Design.** `agents/memory.py` stores rows only once their game's `final_at <= now`, so there is
no leakage. Add a `recall_similar(situation, k)` tool inside A.

**Evaluation.** Tool agent with memory vs without, on the same windows. Measure how often recalled
precedents are cited and whether they change veto precision.

**Effort:** 6–10 hours. **Risk:** low value when there is no edge to remember. **Feasible:** future
work.

---

## 4. Seventy-two-hour plan (Wed 7 → Sat 10 Oct)

### Build these two

1. **Option A: tool-using LLM agent with the sceptic step.** It is the main agentic upgrade, and it
   produces the missing LLM-vs-rules number.
2. **Option B: safety and gate audit.** It is cheap, it closes claims we cannot back up, and the
   placebo test is a rigour highlight.

If two more people are free, they can run **C (simulated-user Coach evaluation)** in parallel. It
is the only route to product evidence. D is a 3–5 hour bonus for one person.

| When | Who (suggested) | Work |
| --- | --- | --- |
| Wed day | Agents subgroup (2) | A: tool wrappers over `AsOf`, provenance IDs in `checks`, `llm.py` function-calling path, response cache |
| Wed day | Evaluation (1–2) | B: kill switch with the `final_at` guard, invariant tests, gate window split and CLV-dollar metric |
| Wed night | All | **Pre-register:** commit hypotheses, prompts, budgets and windows in `docs/prereg_agentic.md` |
| Thu morning | Agents | Prompt freeze on November–January (development); run the leakage probe |
| Thu day | Agents | A runs: test window (main), walk-forward months if the budget allows |
| Thu day | Evaluation | B runs: walk-forward with the fixed gate; 200-rule placebo; paired bootstrap |
| Thu day (optional) | Product (2) | C: Coach graph plus simulator, 30 slates × 3 personas × 2 arms |
| Thu night | All | **Results freeze.** One table: components × evidence gate. Update the deck (2 slides) |
| Fri | Presenters | Dry run; live demo of the tool agent trace on one night plus a blocked order and a kill-switch trip |
| Sat | Report | Write §Agent and §Evaluation; contribution statement; LLM usage statement; submit |

### Future work, named in the report

- full three-role deliberation (F);
- learned full-information policy (G);
- DSL rules (H);
- episodic memory (I);
- full pregame → trader orchestration with calibrated questionable weights (E);
- a human user study for the Coach;
- earlier news sources (injury reports hours before tip, already wired in `pregame.py` live mode)
  re-tested on next season's prices.

### How to present the agent: "The market is the grader"

Frame every component as a hypothesis that must pass an evidence gate on recorded prices. The
same gate applies to models, rules and the LLM.

| Component | Evidence gate | Verdict |
| --- | --- | --- |
| Raw M4 model trading | Beat never trade on CLV and P&L | **Failed** (−$2,475, CI excludes 0) |
| Market anchor | Beat raw model | **Passed** (+$2,450, p = 0.002) |
| Gated rule learning | Beat no learning on CLV $ | **Passed on exposure** (+$39 CLV, p = 0.001); not on per-trade CLV |
| Gate itself | Accept more real rules than placebo rules | *To report (B)* |
| M6 neural impact model | Beat the zero-move martingale | **Failed:** null result, 0 trades |
| Tool-using LLM agent | Beat the deterministic agent and never trade | *To report (A)* |
| Coach | Improve simulated users' CLV and pass rate | *To report (C)* |
| Consumer | Overcome about 1.9¢ all-in cost with information that is 91% priced in already | **Structurally hard:** the thesis |

**Narrative.**
- **Position:** "We built an agent whose job is to *refuse* trades unless evidence survives the
  market's grading."
- **Findings:** "The only components that pass are the ones that defer to the market. Every
  component with more freedom (raw model, neural impact model and, we test, a free LLM agent)
  fails the gate. That finding is the product: if a disciplined agent with our data cannot beat
  the close, a retail user cannot either. So the system teaches (Coach), lets people practise with
  play money (League) and enforces limits (kill switch)."
- **Delivery:** say "we pass" as often as "we trade". Show a trace where the agent passes and
  explains why.

---

## 5. The eight hardest examiner questions

**Q1. "This is a fixed pipeline with an LLM bolted on. Where is the agency?"**

The decide graph is deliberately constrained: code computes every number, and the LLM may only
drop trades. That is a safety choice in a money-moving setting.

To test whether more agency helps, we built a tool-using agent (A). It chooses its own as-of tools
and can propose trades. We compared it with the deterministic agent under identical checks and
caps. [Report the result.]

Agency in our system also lives in the review loop: the agent proposes its own rules, and an
independent gate accepts or rejects them on earlier days only.

**Q2. "Where is the deep learning? Your trades run on logistic regression."**

- **The DL experiments:**
  - **M2:** a GRU minutes and points distribution model with pinball loss.
  - **M6:** ImpactNet, a GRU over the price path plus an MLP, with quantile outputs. We tested
    whether a neural net can predict the market's move to close. Against four baselines, with
    day-clustered CIs, it cannot beat the zero-move martingale.
- **Why M4 stayed logistic:** it was well calibrated and beat neural alternatives on held-out
  Brier score.
- **Why DL is not on the trading path:** we refused to put a model there that failed its gate.
  Putting it there anyway would have been the dishonest choice.

[If G or the Kelly calibration ships, cite it here.]

**Q3. "Nothing beats never trading. Why is this a success?"**

- **What we claimed:** we never claimed an edge. We set out to measure whether one exists for an
  agent with public information.
- **What we found:**
  - The market anchor alone recovers $2,450 over the raw model (p = 0.002).
  - The learning agent sheds $39 of CLV losses compared with no learning (p = 0.001).
  - All-in costs (about 1.9¢ per contract) and information speed (91% of the move happens before
    the inactive list) explain why no setup clears zero.
- **What this proves for consumers:** a disciplined agent with our data and caps cannot beat the
  close, so an unaided retail user is very unlikely to. That is our consumer-fairness thesis, and
  it is why the product is education and play money.

**Q4. "Your learning loop only removes trades. Isn't the gate overfitting?"**

Yes, learning reduces exposure; per-trade CLV is unchanged. We found two weaknesses ourselves:
- the reviewer's selection window overlapped the gate's test window;
- a mean-CLV metric rewards dropping trades.

We fixed both by separating the windows and gating on CLV dollars. We also ran a placebo gate with
random-slice rules. [Report: real acceptance x of 37 vs placebo y of 200.] If real and placebo
acceptance rates match, our claim is limited to "learning cuts losing exposure". We would not
claim "learning discovers lessons".

**Q5. "Your test period isn't clean: the anchor was designed after seeing it."**

Correct, and we disclose it in the README. The honest checks are:
- **Walk-forward:** M4 is retrained before each month, and the notebook starts empty and only uses
  earlier days.
- **Play-off holdout:** scored once, before anyone looked.

The anchor effect holds in the walk-forward (+$2,450, CI +$854 to +$3,984). New components (A, B,
C) were pre-registered on [date] with prompts frozen on November–January.

**Q6. "Can the LLM leak the future, through the data or through its own training?"**

- **Through the data:** all tools wrap the as-of view, which cannot reach settlements. Citations
  after the decision time are rejected by `checks`. The leakage tests delete future rows and check
  that the outputs are unchanged.
- **Through training:** Gemini 2.5 Flash's training cutoff is before the 2025-26 season. We probed
  it on 20 past results and it scored [≈ chance]. A run with anonymised team names gave [similar]
  results.
- **Reproducibility:** responses are cached, so every number in the report regenerates exactly.

**Q7. "Your news is the inactive list at tip−30. Isn't the null result just late data?"**

Partly, and we say so. The market had already moved 91% of the news-direction move before the
list was public. This bounds our claim: Kalshi NBA game-winner prices are efficient *with respect
to public inactive lists*, not with respect to early injury reports.

This is also the consumer-fairness finding. Retail users see news at roughly the time our replay
does, and by then the price has already moved. The pregame agent already ingests the official
injury report, team X accounts and Shams in live mode. Testing it on next season's prices is our
first future-work item.

**Q8. "How do you know the Coach helps anyone, and isn't this a gambling product?"**

- **Evidence so far:** in the simulated-user study (C), coached personas [reduced long-shot and
  chasing trades by x% and raised the pass rate by y%] on identical slates. That shows the
  mechanism, not human learning. Our n = 5 pilot is anecdotal.
- **Harm:**
  - The Coach and League use paper money only, and every screen shows a not-betting-advice and
    problem-gambling notice.
  - The trading mode has code-enforced caps and an enforced daily kill switch.
  - Banned "lock" or "risk-free" wording is blocked by the checks.
  - Media and team channels cannot order at all.
- **Purpose:** the product exists to show users, with their own numbers, that the market is hard
  to beat after costs.
