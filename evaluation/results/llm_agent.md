# LLM tool agent vs deterministic agent vs plain LLM vs never trade

Pre-registration: `docs/preregistration_llm_agent.md`. Same replay as the existing test runs: stake $20, caps $50/$100/$300, fills at the ask (or 1 − bid) plus the Kalshi fee, capped by volume; default M4 (trained before 1 Feb); no learning. CLV is per contract against the mid at tip. 95% CIs and one-sided p-values from a day-clustered bootstrap (2000 replicates, seed 7606).

## test: 2026-02-01 – 2026-04-12 (64 game-days)

**Model deviation from the pre-registration.** The pre-registered `gemini-2.5-flash` is no longer available to new API keys (404), and the free-tier quota for `gemini-3.8-flash` is 20 requests per day. All LLM arms therefore used `gemini-3.5-flash-lite` (free tier, about 15 requests per minute). Prompts, tools, validation and the trading rule are unchanged.

LLM: `gemini-3.5-flash-lite`, temperature 0. Decision points: fixed 40% subsample (seed 7606), same for every setup.

| Setup | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] | p (CLV > 0) |
| --- | --- | --- | --- | --- | --- |
| A. Deterministic anchor agent (no learning) | 53 | -0.0042 [-0.0090, +0.0015] | -19 [-36, -1] | -236 [-538, +73] | 0.959 |
| B. Plain LLM, one call (no tools) | 5 | -0.0030 [-0.0150, +0.0150] | -1 [-4, +2] | -31 [-96, +22] | 0.741 |
| Never trade | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – |

Paired differences on the same game-days (first minus second; p is one-sided for first > second):

| Comparison | Metric | Difference [95% CI] | p |
| --- | --- | --- | --- |
| anchor − never | CLV $ | -19 [-36, -1] | 0.983 |
| anchor − never | P&L | -236 [-538, +73] | 0.932 |
| plain − anchor | mean CLV | +0.0012 [-0.0090, +0.0155] | 0.383 |
| plain − anchor | CLV $ | +18 [+1, +35] | 0.019 |
| plain − anchor | P&L | +206 [-113, +528] | 0.109 |
| plain − never | CLV $ | -1 [-4, +2] | 0.764 |
| plain − never | P&L | -31 [-96, +22] | 0.834 |

Agent diagnostics:

| Setup | Decision points (LLM asked) | Tool calls / decision | Buy proposals | Invalid (rate) | Gap fails | Sceptic rejects / calls | Vetoed vs kept CLV | Trades outside A's games | LLM calls (cache hits) | Tokens in / out | Cost | Latency / call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plain | 304 (304) | 0.0 | 303 | 1 (0.3%) | 298 | – | – | 60% | 304 (224) | 174,816 / 15,773 | $0.02 | 2.80s |

Invalid outputs by reason: plain: {'invalid_json': 1}

Tool use: 

## Tool agent arms (C, D): not run

The tool agent without the sceptic (C) and with the priced-in sceptic (D) were **not scored**. The free-tier daily
quota for `gemini-3.5-flash-lite` (about 500 requests) ran out partway through arm C. On a nested ~100-point
subsample (fraction 0.13 of the same seed, 117 decision points), 32 analyst decisions returned valid output, and all
32 were **pass** (no buy proposals, no invalid outputs). The other 85 failed with HTTP 429 quota errors and placed no
trade. A retry with `gemini-3.5-flash` ran at about 3 requests per minute (high demand), too slow for the deadline.
So H1 (tool agent vs A), H2 (sceptic effect) and the sceptic reject rate have **no result**. H3 is answered only for
the plain LLM (B). To finish once the quota resets (cached calls are free):

    python -m evaluation.llm_agent_eval --window test --backend gemini --model gemini-3.5-flash-lite --subsample 0.4 --setups tool,tool_sceptic --workers 3 --min-interval 12 --report test

**DeepSeek re-run: done on 8 Oct.** See "DeepSeek re-run" below.

**Gemini fallback (7 Oct): pending (API quota).** DeepSeek still returned 402 on 7 Oct. A tool agent with the sceptic (D) run on
`gemini-3.5-flash` over a nested ~53-point subsample (fraction 0.07, seed 7606, nested in the 40% sample) was stopped
after 13 minutes before it scored any decisions: the free tier was too slow for the deadline. No result.

## DeepSeek re-run (8 Oct): B, C and D scored

**Deviations from the pre-registration.** (1) Model: DeepSeek instead of the pre-registered Gemini 2.5 Flash, which
is no longer available to new keys; the Gemini free tier could not score the tool arms. Two DeepSeek models:
`deepseek-chat` (temperature 0, JSON mode) and `deepseek-reasoner`, chosen as the strongest available model that
could finish before the deadline, before seeing its results. On this API both names are served by
DeepSeek-V4.1-Flash; the reasoner has thinking mode on. `deepseek-v4-pro` exists but took about 46 s per call
(about 7 hours for one arm), so it was not run. (2) The same fixed 40% subsample of test decision points as above
(seed 7606, 304 of 771). Prompts, tools, validation, the 4¢ gap after fees, the sceptic, MAX_TOOL_CALLS / MAX_TURNS
and the decision points are identical across both models; nothing was tuned on these results. The anchor (A) is
the same deterministic run.

**Headline.**

- **Neither LLM tool agent beats never trading.** The best arm, the reasoner tool agent with the sceptic (D), made
  2 trades in 304 decisions: CLV $ +$4 [−3, +15], P&L +$100 [−62, +363] vs never trading, p = 0.24.
- **"Beating" the deterministic agent is trading less.** Reasoner D vs A: CLV $ +$23 [+3, +43], p = 0.016, but
  mean CLV per trade +0.0142 [−0.0142, +0.0421], p = 0.24; A loses −$19 [−36, −1] against never trading, and
  D simply avoids those trades. Chat D made 0 trades, so its +$19 [+1, +36] vs A is exactly "never trade vs A".
- **Without the sceptic, the reasoner trades like the anchor.** Reasoner C made 42 trades: mean CLV −0.0040
  [−0.0079, −0.0002], CLV $ −$13 [−30, +5] vs never, no better than A on mean CLV (+0.0002 [−0.0055, +0.0053]).
- **The sceptic helps by vetoing almost everything.** Sceptic effect (D − C), reasoner: CLV $ +$17 [+4, +32],
  p = 0.010; mean CLV +0.0140 [−0.0121, +0.0408], p = 0.23. It rejected 40 of 42 candidate trades (95%); the vetoed
  trades' counterfactual CLV was −0.0047 per contract vs +0.0100 for the 2 kept (n = 2, not evidence of skill).
  Chat: the sceptic rejected 4 of 4; D − C CLV $ +$1 [−0, +2], p = 0.16.
- **The stronger model proposes more trades, not better ones.** Reasoner − chat, tool agent C: CLV $ −$13 [−30, +6],
  mean CLV −0.0015 [−0.0087, +0.0038]; 42 vs 4 trades. The reasoner made 51 buy proposals (chat: 18); none failed
  the 4¢ gap check (chat: 9 did), and 9 failed citation checks. 14% of its decisions were invalid (chat 1.6%),
  mostly empty or malformed JSON: 21 calls used the full 8,192-token budget on hidden reasoning and returned no
  answer, which counts as a pass.
- **Plain LLM on `deepseek-chat` (B):** 3 trades, CLV $ −$1 [−2, +0], P&L +$44 [+0, +101] vs never trading
  (p = 0.054); like the Gemini plain run (B′), it almost never clears the 4¢ gap. On the reasoner, B made 5 trades:
  CLV $ −$0 [−2, +2], P&L +$4 [−77, +75] vs never; reasoner − chat CLV $ +$1 [−0, +2], p = 0.25.

Chat vs reasoner on the same decision points (reasoner minus chat; p one-sided for reasoner > chat):

| Setup | Trades (test-deepseek-reasoner / test-deepseek) | Metric | test-deepseek-reasoner − test-deepseek [95% CI] | p |
| --- | --- | --- | --- | --- |
| plain | 5 / 3 | mean CLV | +0.0053 [+0.0000, +0.0150] | 0.109 |
| plain | 5 / 3 | CLV $ | +1 [-0, +2] | 0.250 |
| plain | 5 / 3 | P&L | -40 [-102, +0] | 0.918 |
| tool | 42 / 4 | mean CLV | -0.0015 [-0.0087, +0.0038] | 0.715 |
| tool | 42 / 4 | CLV $ | -13 [-30, +6] | 0.932 |
| tool | 42 / 4 | P&L | -51 [-465, +443] | 0.565 |
| tool_sceptic | 2 / 0 | mean CLV | +0.0100 [-0.0150, +0.0350] | 0.259 |
| tool_sceptic | 2 / 0 | CLV $ | +4 [-3, +15] | 0.236 |
| tool_sceptic | 2 / 0 | P&L | +100 [-62, +363] | 0.236 |

### test-deepseek: 2026-02-01 – 2026-04-12 (64 game-days)

**Model deviation from the pre-registration.** The pre-registered model was Gemini 2.5 Flash (`gemini-2.5-flash`), which is no longer available to new API keys, and the Gemini free-tier quota ran out before the tool arms could be scored. Setups B, C and D therefore all used DeepSeek `deepseek-chat` (temperature 0, JSON mode), so every LLM setup in the main comparison uses the same model. Prompts, tools, validation, the decision points and the trading rule are unchanged. The anchor (A) is the same deterministic run as before. The earlier plain-LLM run on `gemini-3.5-flash-lite` is kept as a secondary row (B′) for reference.

LLM: `deepseek-chat`, temperature 0. Decision points: fixed 40% subsample (seed 7606), same for every setup.

| Setup | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] | p (CLV > 0) |
| --- | --- | --- | --- | --- | --- |
| A. Deterministic anchor agent (no learning) | 53 | -0.0042 [-0.0090, +0.0015] | -19 [-36, -1] | -236 [-538, +73] | 0.959 |
| B. Plain LLM, one call (no tools) (`deepseek-chat`) | 3 | -0.0083 [-0.0150, -0.0050] | -1 [-2, +0] | +44 [+0, +101] | 1.000 |
| C. Tool agent, no sceptic (`deepseek-chat`) | 4 | -0.0025 [-0.0050, +0.0050] | -1 [-2, +0] | -42 [-123, +34] | 1.000 |
| D. Tool agent + priced-in sceptic (`deepseek-chat`) | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – |
| B′. Plain LLM, one call (no tools), earlier run (`gemini-3.5-flash-lite`) | 5 | -0.0030 [-0.0150, +0.0150] | -1 [-4, +2] | -31 [-96, +22] | 0.741 |
| Never trade | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – |

Paired differences on the same game-days (first minus second; p is one-sided for first > second):

| Comparison | Metric | Difference [95% CI] | p |
| --- | --- | --- | --- |
| anchor − never | CLV $ | -19 [-36, -1] | 0.983 |
| anchor − never | P&L | -236 [-538, +73] | 0.932 |
| plain − anchor | mean CLV | -0.0041 [-0.0119, +0.0031] | 0.885 |
| plain − anchor | CLV $ | +18 [+1, +35] | 0.019 |
| plain − anchor | P&L | +280 [-48, +595] | 0.048 |
| plain − never | CLV $ | -1 [-2, +0] | 0.931 |
| plain − never | P&L | +44 [+0, +101] | 0.054 |
| tool − anchor | mean CLV | +0.0017 [-0.0054, +0.0094] | 0.277 |
| tool − anchor | CLV $ | +18 [-0, +36] | 0.021 |
| tool − anchor | P&L | +194 [-127, +504] | 0.103 |
| tool − never | CLV $ | -1 [-2, +0] | 0.846 |
| tool − never | P&L | -42 [-123, +34] | 0.888 |
| tool_sceptic − anchor | mean CLV | +0.0042 [-0.0015, +0.0090] | 0.041 |
| tool_sceptic − anchor | CLV $ | +19 [+1, +36] | 0.018 |
| tool_sceptic − anchor | P&L | +236 [-73, +538] | 0.069 |
| tool_sceptic − never | CLV $ | +0 [+0, +0] | – |
| tool_sceptic − never | P&L | +0 [+0, +0] | – |
| plain_gemini − anchor | mean CLV | +0.0012 [-0.0090, +0.0155] | 0.383 |
| plain_gemini − anchor | CLV $ | +18 [+1, +35] | 0.019 |
| plain_gemini − anchor | P&L | +206 [-113, +528] | 0.109 |
| plain_gemini − never | CLV $ | -1 [-4, +2] | 0.764 |
| plain_gemini − never | P&L | -31 [-96, +22] | 0.834 |
| tool_sceptic − tool | mean CLV | +0.0025 [-0.0050, +0.0050] | 0.358 |
| tool_sceptic − tool | CLV $ | +1 [-0, +2] | 0.156 |
| tool_sceptic − tool | P&L | +42 [-34, +123] | 0.113 |

Agent diagnostics:

| Setup | Decision points (LLM asked) | Tool calls / decision | Buy proposals | Invalid (rate) | Gap fails | Sceptic rejects / calls | Vetoed vs kept CLV | Trades outside A's games | LLM calls (cache hits) | Tokens in / out | Cost | Latency / call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plain | 304 (304) | 0.0 | 304 | 0 (0.0%) | 301 | – | – | 0% | 304 (4) | 168,709 / 21,378 | $0.07 | 1.90s |
| tool | 304 (304) | 7.2 | 18 | 5 (1.6%) | 9 | – | – | 25% | 815 (815) | 955,851 / 132,103 | $0.40 | 1.97s |
| tool_sceptic | 304 (304) | 7.2 | 18 | 5 (1.6%) | 9 | 4 / 4 (100%) | -0.0025 vs +nan | – | 819 (11) | 963,475 / 132,508 | $0.41 | 1.97s |
| plain_gemini | 304 (304) | 0.0 | 303 | 1 (0.3%) | 298 | – | – | 60% | 304 (224) | 174,816 / 15,773 | $0.02 | 2.80s |

Invalid outputs by reason: plain: none; tool: {'ungrounded_estimate': 3, 'number_mismatch': 2}; tool_sceptic: {'ungrounded_estimate': 3, 'number_mismatch': 2}; plain_gemini: {'invalid_json': 1}

Tool use: tool: {'get_quote': 608, 'get_anchor': 350, 'get_news': 304, 'get_rotation': 261, 'm6_predicted_move': 251, 'edge': 201, 'm4_win_prob': 200}; tool_sceptic: {'get_quote': 608, 'get_anchor': 350, 'get_news': 304, 'get_rotation': 261, 'm6_predicted_move': 251, 'edge': 201, 'm4_win_prob': 200}

### test-deepseek-reasoner: 2026-02-01 – 2026-04-12 (64 game-days)

**Stronger model (deviation, chosen before seeing its results).** Setups B, C and D re-run on DeepSeek `deepseek-reasoner`, chosen as the strongest available model that could finish before the deadline, before seeing its results. On this API both `deepseek-chat` and `deepseek-reasoner` are served by DeepSeek-V4.1-Flash; the reasoner has thinking mode on (hidden chain of thought, kept out of the parsed answer). Temperature is not sent (thinking mode ignores it), max_tokens is 8192; replies are cached, so the replay is fixed. Prompts, tools, validation, the 4¢ edge rule, the sceptic, MAX_TOOL_CALLS / MAX_TURNS and the decision points are identical to the `deepseek-chat` run.

LLM: `deepseek-reasoner`, thinking mode. Decision points: fixed 40% subsample (seed 7606), same for every setup.

| Setup | Trades | Mean CLV [95% CI] | CLV $ [95% CI] | P&L after fees [95% CI] | p (CLV > 0) |
| --- | --- | --- | --- | --- | --- |
| A. Deterministic anchor agent (no learning) | 53 | -0.0042 [-0.0090, +0.0015] | -19 [-36, -1] | -236 [-538, +73] | 0.959 |
| B. Plain LLM, one call (no tools) | 5 | -0.0030 [-0.0117, +0.0083] | -0 [-2, +2] | +4 [-77, +75] | 0.739 |
| C. Tool agent, no sceptic | 42 | -0.0040 [-0.0079, -0.0002] | -13 [-30, +5] | -93 [-492, +400] | 0.982 |
| D. Tool agent + priced-in sceptic | 2 | +0.0100 [-0.0150, +0.0350] | +4 [-3, +15] | +100 [-62, +363] | 0.298 |
| Never trade | 0 | – | +0 [+0, +0] | +0 [+0, +0] | – |

Paired differences on the same game-days (first minus second; p is one-sided for first > second):

| Comparison | Metric | Difference [95% CI] | p |
| --- | --- | --- | --- |
| anchor − never | CLV $ | -19 [-36, -1] | 0.983 |
| anchor − never | P&L | -236 [-538, +73] | 0.932 |
| plain − anchor | mean CLV | +0.0012 [-0.0074, +0.0114] | 0.374 |
| plain − anchor | CLV $ | +19 [+2, +35] | 0.016 |
| plain − anchor | P&L | +240 [-88, +564] | 0.075 |
| plain − never | CLV $ | -0 [-2, +2] | 0.576 |
| plain − never | P&L | +4 [-77, +75] | 0.452 |
| tool − anchor | mean CLV | +0.0002 [-0.0055, +0.0053] | 0.453 |
| tool − anchor | CLV $ | +5 [-13, +25] | 0.260 |
| tool − anchor | P&L | +143 [-298, +633] | 0.256 |
| tool − never | CLV $ | -13 [-30, +5] | 0.941 |
| tool − never | P&L | -93 [-492, +400] | 0.643 |
| tool_sceptic − anchor | mean CLV | +0.0142 [-0.0142, +0.0421] | 0.241 |
| tool_sceptic − anchor | CLV $ | +23 [+3, +43] | 0.016 |
| tool_sceptic − anchor | P&L | +337 [-27, +724] | 0.045 |
| tool_sceptic − never | CLV $ | +4 [-3, +15] | 0.236 |
| tool_sceptic − never | P&L | +100 [-62, +363] | 0.236 |
| tool_sceptic − tool | mean CLV | +0.0140 [-0.0121, +0.0408] | 0.229 |
| tool_sceptic − tool | CLV $ | +17 [+4, +32] | 0.010 |
| tool_sceptic − tool | P&L | +194 [-187, +517] | 0.145 |

Agent diagnostics:

| Setup | Decision points (LLM asked) | Tool calls / decision | Buy proposals | Invalid (rate) | Gap fails | Sceptic rejects / calls | Vetoed vs kept CLV | Trades outside A's games | LLM calls (cache hits) | Tokens in / out | Cost | Latency / call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plain | 304 (304) | 0.0 | 299 | 5 (1.6%) | 294 | – | – | 40% | 304 (0) | 176,309 / 809,786 | $0.94 | 14.40s |
| tool | 299 (299) | 7.8 | 51 | 42 (14.0%) | 0 | – | – | 48% | 1,068 (1,068) | 1,230,973 / 1,447,344 | $1.92 | 7.17s |
| tool_sceptic | 304 (304) | 7.8 | 51 | 43 (14.1%) | 0 | 40 / 42 (95%) | -0.0047 vs +0.0100 | 50% | 1,129 (0) | 1,311,595 / 1,578,305 | $2.09 | 7.41s |

Invalid outputs by reason: plain: {'invalid_json': 5}; tool: {'invalid_json': 16, 'bad_schema': 16, 'number_mismatch': 8, 'budget_exhausted': 1, 'ungrounded_estimate': 1}; tool_sceptic: {'invalid_json': 16, 'bad_schema': 16, 'number_mismatch': 8, 'budget_exhausted': 2, 'ungrounded_estimate': 1}

Tool use: tool: {'get_quote': 598, 'get_anchor': 590, 'get_news': 299, 'edge': 275, 'm6_predicted_move': 265, 'm4_win_prob': 259, 'get_rotation': 41, 'get_price_history': 13, 'fee': 2, 'breakeven': 1}; tool_sceptic: {'get_quote': 608, 'get_anchor': 600, 'get_news': 304, 'edge': 279, 'm6_predicted_move': 270, 'm4_win_prob': 264, 'get_rotation': 41, 'get_price_history': 13, 'fee': 2, 'breakeven': 1}

Hidden reasoning (thinking mode, not shown to the parser): plain: 789,571 reasoning tokens, 9,784 characters per call; tool: 1,310,232 reasoning tokens, 4,162 characters per call; tool_sceptic: 1,435,444 reasoning tokens, 4,357 characters per call

**Cost and latency (DeepSeek).** Actual spend from the account balance: $40.00 before, $36.51 after = **$3.49 for
all six DeepSeek arms plus a few probe calls** (one to `deepseek-v4-pro`). Token-based estimates at
`agents/llm_client.PRICES` ($0.27 / $1.10 per million, both models) agree: chat B $0.07, chat D $0.41 (C replays
D's analyst calls from the cache), so **chat ≈ $0.48**; reasoner D $2.09 (C replays D), reasoner B $0.94, so
**reasoner ≈ $3.03**. The reasoner is slower per call (tool agent 7.4 s vs 2.0 s; plain 14.4 s vs 1.9 s) and writes
hidden reasoning of about 4,300 characters per tool-agent call and 9,800 per plain call. Latency is per API call;
the runs used 3–6 parallel workers.

**Future work.** Raise `max_tokens` for the reasoner (or `MAX_TOOL_CALLS` / `MAX_TURNS`) and try
`deepseek-v4-pro`, on new days only: both were kept fixed here for comparability and to avoid tuning on test.

![Cumulative P&L and CLV by setup: Gemini test run, DeepSeek chat, DeepSeek reasoner](llm_agent.png)

Costs are estimates from token counts at the list prices in `agents/llm_client.PRICES` (Gemini 3.5 Flash-Lite assumed at the 2.5 Flash-Lite price, $0.10 / $0.40 per million input / output tokens; the runs themselves used the free tier) and include calls served from the cache, i.e. the cost of a fresh run. Latency is per API call as measured when the response was first fetched.
