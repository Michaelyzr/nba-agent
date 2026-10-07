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

### 2. The problem: surging, under-protected betting markets (0:55) [1:10]

> First, volume is surging. Monthly trading on Kalshi and Polymarket more than doubled, from
> $26 billion in May to $53 billion in July 2026, mostly sports, and Kalshi did a record $61
> billion in September, about 46 times a year earlier. It reaches ordinary users through apps:
> nearly 2 million Robinhood customers, and Coinbase. Second, price protection is thin. An exchange
> has no house edge, but you pay a fee and the spread, and you trade against professional market
> makers, including Kalshi's own affiliate. The CFTC polices market integrity, not whether your
> price is fair, and not gambling harm; you can trade at 18. States are fighting it in court, and
> the courts are split. Our own data shows what that gap costs: about 5% per trade, and 91% of the
> move before the public news. Third, betting causes measurable harm. Consumers need education and
> protection, not tips.

*Sources (say only if asked): Pew Research 23 Sep 2026; TickerTracker via NEXTPredict;
Robinhood Q2 2026; 3rd Cir. 6 Apr 2026 and 9th Cir. 28 Aug 2026. Full list in
`docs/market_regulation.md`. Notional volume counts each contract at $1, so cash paid is lower.*

### 3. Research question and what we built (0:30) [1:40]

> Our question: can a consumer, even one armed with an AI agent and public news, beat the market?
> We built the strongest consumer we could: public news, trained models including a neural net,
> a LangGraph agent with code risk limits, gated learning and LLM tools. A replay shows it only
> what was public and fills every order at the real ask plus fee. The answer is no.

### 4. How the agent works (0:45) [2:25]

> Here is the agent. Top row, the decide loop: a news or clock trigger wakes it; it picks which
> as-of tools to call, prices, news, rotations, our models; it estimates a probability, market
> anchor plus model shift; it trades only if the edge after fees is at least 4 cents, an LLM
> sceptic can veto "already priced in", and code checks, risk caps and a kill switch the LLM can't
> override come last. Example: CLE at Portland, Avdija out, 26 points a game. The model shifts
> 8.7 points, 42.5% becomes 33.8%; it buys Portland "no" at 55 cents, 1.73 cents fee, 9.5 cents
> of edge, 36 contracts. The close is 62.5: plus 7.5 cents of CLV, plus $15.57. Bottom row: every
> trade is graded at the close, a reviewer proposes a rule, and the gate keeps it only if it helps
> on held-out days. Coach and League give the same analysis to people. The close grades every box.

### 5. Market-Graded Learning: the closing price is the grader (0:35) [3:00]

> This is what the title means. Each component must pass the market's test. The raw win model
> fails: minus $2,475 over the season. Recalibrating it fails too. The neural impact model fails:
> all 9 configurations are worse than "the price won't move". What passes defers to the market:
> the market anchor, plus $2,450, p 0.002, and gated learning, which passes only by trading less.
> So the finding is the product: if a disciplined agent can't beat the close, a consumer can't.

### 6. Data (0:15) [3:15]

> All public: 3.9 million one-minute Kalshi price rows, and ESPN box scores and inactive lists as
> our news. Test is February to April, 501 games; the 87 play-off games are a holdout scored once.
> No look-ahead, and every fill pays ask plus fee.

### 7. Evidence 1: costs, and a price that moves before the news (0:45) [4:00]

> Costs first. A $20 order pays 1.92 cents per contract, 5% of the price, up to 14% on cheap
> contracts. Our best agent made plus $36 at the mid and lost $32 after costs. Second, speed: 91%
> of the move happens before the inactive list, and about two-thirds, 67%, before the first
> official injury report, about 6 hours earlier. That is why the agent passes by default.

### 8. Evidence 2: the market beats our model (0:10) [4:10]

> Third, expertise: Brier score market 0.164, our model 0.186, and when we disagree the market is
> right.

### 9. Deep learning: four neural parts, graded by the market (0:45) [4:55]

> Deep learning sits in four places. The M2 player GRU beats gradient boosting, pinball 1.55
> against 1.59. The neural win model, M4-NN, ties logistic regression, Brier 0.184 against 0.183.
> With 3,400 games, extra capacity doesn't pay. But as the news shift on the market anchor it
> reaches 0.163, level with the market's 0.164. M6, a GRU over six hours of prices with quantile
> outputs, is graded by the market directly, and all 9 configurations are worse than "the price
> won't move". In an efficient market that null is the finding. The LLM agent is the fourth
> neural part: a transformer used as a policy with tools.

### 10. Evidence 3: money over time, tested with CIs (0:35) [5:30]

> Money after fees; zero is never trading. The raw model falls steadily, and that slope is costs.
> Learning flattens the line by trading less. Walk-forward with bootstrap intervals: closing-line
> value is below zero for every setup; the full agent's minus $25 is indistinguishable from never
> trading. The anchor is the one real effect. In the play-offs the full agent made no trades.

### 11. Agentic upgrades (0:45) [6:15]

> We graded each agentic upgrade the same way. The new gate kept 5 of 11 rules and cut trades
> from 129 to 53: plus $18 of CLV, no better per trade, and never trading still wins. Random
> placebo rules pass it just as often. The Coach raises simulated users' CLV, up to $88, by
> cutting their trades. The plain LLM made 5 trades in 304, the same as never trading.
> [Tool agent: "pending, re-running on DeepSeek" unless `K["llm_tool"]` is filled.] Every
> upgrade, graded by the market, converges on the same answer: trade less. The market is hard to
> beat after fees, which is exactly why the product is education, not a trading bot.

### 12. Our product: protect, teach, practise (0:30) [6:45]

> So the product protects, teaches and lets people practise. The agent passes by default, 90% of
> the time, and says why; limits are code the LLM can't override; 9 of 9 planted violations were
> blocked. The Coach shows the break-even after costs. The League ranks on closing-line value, not
> profit: our raw-model bot is up $15 with negative CLV, and the badge says "costs".

### 13. Why an agentic framework, and who built which part (0:35) [7:20]

> Why an agent and not one classifier? Each box on slide 4 answers the problem. *When* to act is
> the trigger: 91% of the move comes before the inactive list. *What* to look at is the as-of
> tools, with no look-ahead. *Whether* to trade is the edge test and sceptic, behind code limits
> the LLM can't override. And our data answers "a classifier would do": the model alone loses
> $2,475 over 711 trades,
> and the decision process around it adds $2,450, p 0.002. The team built the pregame news loop, PR 7; the player-feature experiment, PR 8; in-play
> updates, PR 9; and live Polymarket prices, PRs 11 and 12. The replay, agent, models and
> evaluation are on our main branch. Each part faced the same market grade.

### 14. Live demo (2:00) [9:20]

- Replayed night: news, then estimate against bid and ask, then usually no order, with the reason.
- Coach: one paper call, compared with the close and the agent.
- League: one call on the shared slate, close revealed, leaderboard ranked by CLV.
- Safety: a planted violation blocked.
- If anything fails, switch to the backup recording.

### 15. Recommendations, honest scope and Q&A (0:30) [9:50]

> Show the all-in cost and break-even before every order; warn when news is likely priced in; put
> education and play-money practice before the first trade. Honest scope: unfair means costs,
> speed and expertise, not fraud, and the walk-forward and play-offs are our honest tests. Thank
> you, we're happy to take questions.

---

**Buffer:** about 10 seconds. If running long, drop the valuation line on slide 2, or skip the
League call in the demo. Backup figures for Q&A: `figures/walkforward.png`,
`figures/data_timeline.png`, `figures/longshot_calibration.png`.
