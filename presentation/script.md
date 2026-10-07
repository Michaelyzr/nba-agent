# Rehearsal script (about 10 minutes, Fri 9 Oct 2026)

Deck: `nba_agent_deck.pptx` (15 slides). Times are targets; the running total is in brackets.
Speak at about 150 words a minute. The speaker notes in the deck carry the same content in more
detail. Numbers come from `K` in `build_deck.py`; if a number changes there, change it here too.

**Delivery rule:** say "we pass" as often as "we trade". Every component is graded by the market's
closing price.

---

### 1. Title: Market-Graded Learning (0:15) [0:15]

> Good morning, we are Group 12. Our title, Market-Graded Learning, is also our thesis. We built an
> AI agent that trades NBA prediction markets and let the market's closing price grade every part
> of it. Only the parts that defer to the market survive.

### 2. The problem (0:35) [0:50]

> Consumers face three things. Betting is mainstream: legal in 38 US states, and 46% of adults
> worldwide gambled last year. The markets are tilted, which is what we measured: about 5% cost per
> trade, and 91% of the price move before the public news. And published research shows harm:
> lower credit scores, more bankruptcies. Consumers need education and protection, not tips.

### 3. Research question and what we built (0:30) [1:20]

> Our question: can a consumer, even one armed with an AI agent and public news, beat the market?
> We built the strongest consumer we could: public news, trained models including a neural net,
> a LangGraph agent with code risk limits, gated learning and LLM tools. A replay shows it only
> what was public and fills every order at the real ask plus fee. The answer is no.

### 4. Market-Graded Learning: the closing price is the grader (0:40) [2:00]

> This is what the title means. Each component must pass the market's test. The raw win model
> fails: minus $2,475 over the season. Recalibrating it fails too. The neural impact model fails:
> all 9 configurations are worse than "the price won't move". What passes defers to the market:
> the market anchor, plus $2,450, p 0.002, and gated learning, which passes only by trading less.
> So the finding is the product: if a disciplined agent can't beat the close, a consumer can't.

### 5. Data (0:30) [2:30]

> All public. Kalshi: 2,632 game-winner markets, 3.9 million one-minute price rows, settlements.
> ESPN: three seasons of games, box scores and inactive lists, which are our news. Models train
> before February; the test is February to April, 501 games; the walk-forward retrains monthly;
> the 87 play-off games are a holdout scored once. No look-ahead, and every fill pays ask plus fee.

### 6. Evidence 1: costs (0:30) [3:00]

> Costs first. A $20 order pays 1.92 cents per contract, 5% of the price, up to 14% on cheap
> contracts. Our best agent made plus $36 at the mid and lost $32 after costs. Even trading 74
> times out of 771 chances, only about a third beat the closing price.

### 7. Evidence 2: information speed (0:35) [3:35]

> Second, speed. 91% of the move happens before the inactive list, when a retail user could act.
> We then matched the official NBA injury reports: about two-thirds of the move, 67%, comes before
> the first official report, which itself arrives about 6 hours before the inactive list. Even
> assuming no publication lag it is about half. And those are official reports only.

### 8. Evidence 3: the market beats our model (0:20) [3:55]

> Third, expertise. Brier score one hour before tip: market 0.164, our model 0.186. When we
> disagree by more than five points, the market is right in both directions.

### 9. Deep learning: four neural parts, graded by the market (0:45) [4:40]

> Deep learning sits in four places. The M2 player GRU beats gradient boosting, pinball 1.55
> against 1.59. The neural win model, M4-NN, ties logistic regression, Brier 0.184 against 0.183.
> With 3,400 games, extra capacity doesn't pay. But as the news shift on the market anchor it
> reaches 0.163, level with the market's 0.164. M6, a GRU over six hours of prices with quantile
> outputs, is graded by the market directly, and all 9 configurations are worse than "the price
> won't move". In an efficient market that null is the finding. The LLM agent is the fourth
> neural part: a transformer used as a policy with tools.

### 10. Evidence 4: money over time, tested with CIs (0:40) [5:20]

> Money after fees; zero is never trading. The raw model falls steadily, and that slope is costs.
> Learning flattens the line by trading less. Walk-forward with bootstrap intervals: closing-line
> value is below zero for every setup; the full agent's minus $25 is indistinguishable from never
> trading. The anchor is the one real effect. In the play-offs the full agent made no trades.

### 11. Agentic upgrades (0:45) [6:05]

> We graded each agentic upgrade the same way. The new gate kept 5 of 11 rules and cut trades
> from 129 to 53: plus $18 of CLV, no better per trade, and never trading still wins. Random
> placebo rules pass it just as often. The Coach raises simulated users' CLV, up to $88, by
> cutting their trades. The plain LLM made 5 trades in 304, the same as never trading.
> [Tool agent: "pending, re-running on DeepSeek" unless `K["llm_tool"]` is filled.] Every
> upgrade, graded by the market, converges on the same answer: trade less. The market is hard to
> beat after fees, which is exactly why the product is education, not a trading bot.

### 12. Our product: protect, teach, practise (0:35) [6:40]

> So the product protects, teaches and lets people practise. The agent passes by default, 90% of
> the time, and says why; limits are code the LLM can't override; 9 of 9 planted violations were
> blocked. The Coach shows the break-even after costs. The League ranks on closing-line value, not
> profit: our raw-model bot is up $15 with negative CLV, and the badge says "costs".

### 13. Why an agentic framework, and who built which part (0:35) [7:15]

> Why an agent and not one classifier? The hard decision is *when* to act: 91% of the move comes
> before the inactive list, two-thirds before the first official report. The agent's trigger,
> investigate, decide loop makes that call, with as-of tools, code risk caps and a kill switch.
> And our data answers "a classifier would do": the model alone loses $2,475 over 711 trades,
> and the decision process around it adds $2,450, p 0.002. The team built the pregame news loop, PR 7; the player-feature experiment, PR 8; in-play
> updates, PR 9; and live Polymarket prices, PRs 11 and 12. The replay, agent, models and
> evaluation are on our main branch. Each part faced the same market grade.

### 14. Live demo (2:00) [9:15]

- Replayed night: news, then estimate against bid and ask, then usually no order, with the reason.
- Coach: one paper call, compared with the close and the agent.
- League: one call on the shared slate, close revealed, leaderboard ranked by CLV.
- Safety: a planted violation blocked.
- If anything fails, switch to the backup recording.

### 15. Recommendations, honest scope and Q&A (0:30) [9:45]

> Show the all-in cost and break-even before every order; warn when news is likely priced in; put
> education and play-money practice before the first trade. Honest scope: unfair means costs,
> speed and expertise, not fraud, and the walk-forward and play-offs are our honest tests. Thank
> you, we're happy to take questions.

---

**Buffer:** about 15 seconds. If running long, shorten slide 5 to one sentence or skip the
League call in the demo. Backup figures for Q&A: `figures/walkforward.png`,
`figures/data_timeline.png`, `figures/longshot_calibration.png`.
