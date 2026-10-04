# NBA Pre-game Spread Analysis Specification

## Purpose

The central question in NBA spread analysis is not simply which team is more
likely to win. It is whether the model's distribution of the final scoring
margin differs meaningfully from the market spread after accounting for
injuries, lineup uncertainty, market movement, pricing and model error.

For example, if the home team is listed at -5.5 and the model projects a mean
home margin of 8.0 points, the raw model-versus-market difference is 2.5
points. That difference is not sufficient by itself. The system must also
estimate the probability distribution of the margin, check whether important
news has already been priced into the market, and require the advantage to be
large enough to survive forecast uncertainty and transaction costs.

This document is a human-authored analysis specification. It defines the
features, checks and workflow that the forecasting agent should use. It is not
a list of automatically approved trading rules. Machine-executable learned
rules belong in `notebook.json` only after they pass the rule gate.

## Core quantities

Use a consistent sign convention throughout the system:

- `predicted_home_margin`: expected home points minus expected away points.
- `market_home_margin`: the market-implied home margin. A home spread of -5.5
  corresponds to a market home margin of +5.5 points.
- `margin_edge`: `predicted_home_margin - market_home_margin`.
- `cover_probability`: probability that the selected side covers the current
  spread, including push handling for integer lines.
- `fair_probability`: implied probability after removing the bookmaker margin
  or exchange fees.
- `probability_edge`: `cover_probability - fair_probability`.

A positive margin edge does not guarantee a profitable decision. The system
must use cover probability and uncertainty, not only a point estimate.

## 1. Team strength

Do not use win-loss record or points per game as the primary strength signal.
Estimate performance on a per-possession basis and include:

- Offensive Rating.
- Defensive Rating.
- Net Rating.
- Home and away Net Rating.
- Performance over the most recent 5, 10 and 15 games.
- Opponent-adjusted performance.
- Differences between long-term season strength and recent form.

A simple starting estimate is:

```text
expected margin = home team strength - away team strength + home-court adjustment
```

Net Rating is generally more informative than points per game because it
controls for pace. Recent-form windows must never include games completed
after the forecast's `as_of` timestamp.

## 2. Injuries and expected lineups

The forecast must distinguish a confirmed absence from an uncertain status.
Collect and model:

- Official injury status and the time it became available.
- Probability that a questionable or doubtful player participates.
- Team Net Rating with and without the affected player.
- Expected starters and rotation changes.
- Expected minutes and any minutes restriction.
- Whether replacement players can absorb the same position, minutes, usage
  and playmaking responsibility.
- Expected effects on offense, defense, rebounding and creation.

Do not translate every star absence into a fixed point adjustment. Player
impact depends on role, teammates, opponent and replacement depth. Every
injury adjustment must record its source and `available_at` timestamp.

## 3. Pace and scoring environment

Estimate the expected number of possessions and the efficiency environment.
Relevant features include:

- Pace and expected possessions.
- Offensive and Defensive Rating.
- Three-point attempt rate and three-point percentage.
- Free-throw rate.
- Turnover rate.
- Offensive rebound rate.
- Opponent rim, transition, rebounding and three-point defense.

Pace affects both the expected margin and the variance of the margin. More
possessions can create more opportunity for a stronger team to separate, but
the interaction must be estimated from historical data rather than applied as
an unconditional rule.

## 4. Matchup effects

Historical head-to-head record alone is not a sufficient matchup feature.
Describe how the teams' styles interact:

- Pick-and-roll offense versus pick-and-roll defense.
- Interior scoring versus rim protection.
- Three-point creation versus opponent three-point suppression.
- Transition offense versus transition defense.
- Offensive rebounding versus defensive rebounding.
- Isolation and mismatch creation.
- Switch-heavy, drop, zone, small-ball and large-lineup strategies.
- Whether an opponent can repeatedly target a specific defender or coverage.

Matchup adjustments should be bounded and validated on held-out games. They
must not become unrestricted narrative overrides of the statistical model.

## 5. Schedule, travel and fatigue

Check:

- Back-to-back status.
- Games played in the previous 7 and 14 days.
- Consecutive road games.
- Travel distance and time-zone changes.
- Overtime in the previous game.
- Rest-day difference between the teams.
- Late-season, pre-playoff, tanking or likely-rest situations.
- Team-specific rest and rotation tendencies.

Fatigue can affect defensive effort, transition defense and rebounding as well
as shooting. Schedule features should therefore be allowed to affect both the
mean and variance of the forecast.

## 6. Home-court effects

Do not apply the same fixed home-court value to every team and season. Consider:

- Team-specific home and away Net Rating.
- Home and away shooting splits, with shrinkage for small samples.
- Opponent road-offense decline.
- Travel distance, time zone and altitude.
- Arena-specific effects when supported by sufficient data.
- Free-throw and officiating differences only when measured without leakage.

Home-court estimates should be regularized toward a league-wide prior so that
small samples do not produce extreme adjustments.

## 7. Spread and market data

For every observation, record:

- Opening spread and timestamp.
- Current spread and timestamp.
- Opening and current price or odds for both sides.
- Total and total movement.
- Quotes from each available sportsbook or exchange.
- Market consensus or a documented aggregation method.
- The exact time of every line and price change.

Differentiate between:

- A spread move, such as -4.5 to -5.5.
- A price move while the spread remains unchanged.
- Rapid repricing after an injury update.

Public bet percentages should not be treated as a direct signal unless the
source, sampling method and historical usefulness are documented. Market data
must be filtered to the decision timestamp.

## 8. Motivation and game context

Possible context features include:

- Playoff-seeding incentives.
- Clinched standings and likely rest.
- Repeated games against the same opponent.
- Coaching experiments and unusual rotations.
- Late-season organizational incentives.
- Recent trades or temporary roster disruption.

These features are difficult to quantify. Record them as structured context
with a source, timestamp and confidence level. Do not allow unsupported
narrative claims to override the model.

## 9. Margin uncertainty

The system must output a distribution rather than only one predicted margin.
Important sources of uncertainty include:

- Three-point shooting variance.
- Free-throw volume.
- Turnovers and offensive rebounds.
- Garbage-time scoring.
- Intentional fouling at the end of the game.
- Pace reduction by a leading team.
- Late scratches and unexpected minutes restrictions.
- Low-probability blowouts and upsets.

A prediction of a six-point home win means the center of the projected margin
distribution is near six points. It does not mean six is the uniquely most
likely final margin. Report at least a mean or median, prediction intervals and
cover probability.

## 10. Odds and cover probability

Convert the quoted odds into implied probabilities and remove the bookmaker
margin when possible. An actionable research signal requires:

```text
model cover probability > fair market probability
```

and the difference must exceed a pre-registered threshold that accounts for:

- Calibration error.
- Sampling uncertainty.
- Quote staleness.
- Spread and price movement.
- Exchange fees or bookmaker margin.
- Model-selection and multiple-testing risk.

For example:

```text
model cover probability: 55.0%
fair market probability:  52.4%
raw probability edge:      2.6 percentage points
```

This describes a possible long-run expected-value advantage, not a guarantee
for a single game. The initial system should evaluate paper decisions only.

## Minimum game-level dataset

Store at least the following fields for each forecast snapshot:

```text
game_id
as_of
game_date
home_team_id
away_team_id
tip_time
opening_spread
current_spread
opening_home_price
current_home_price
market_source
home_net_rating
away_net_rating
home_away_split_ratings
recent_net_ratings_5_10_15
opponent_adjusted_ratings
expected_starters
injury_statuses
injury_available_at
expected_minutes
pace
offensive_ratings
defensive_ratings
rest_days
games_in_previous_7_days
games_in_previous_14_days
travel_distance
time_zone_change
total_line
predicted_home_margin
margin_interval
cover_probability
model_version
feature_version
final_home_score
final_away_score
final_margin
covered
```

Every feature used for a decision must be reproducible from data available at
or before `as_of`.

## Reference analysis workflow

1. Build an initial margin estimate from season-long, opponent-adjusted team
   strength.
2. Blend long-term strength with recent form using validated weights.
3. Apply team-specific home-court, rest, schedule and travel adjustments.
4. Estimate player availability, expected lineups and minutes.
5. Recompute offense, defense and rebounding expectations for the projected
   rotations.
6. Apply bounded, testable matchup features.
7. Predict pace, scoring efficiency and the full margin distribution.
8. Compare the distribution with the opening and current spread and prices.
9. Check whether injury news and other information appear to have already been
   incorporated into the market.
10. Compute cover probability, fair market probability and uncertainty-aware
    edge.
11. Run leakage, freshness, policy and risk checks.
12. Save the complete pre-game forecast and evaluate it after settlement.

## Data-source boundaries

NBA statistics can supply much of the team, player, lineup, schedule and box
score information. They do not by themselves provide a complete spread
analysis dataset. The pipeline also needs timestamped injury reports and a
separate market-data source for opening/current spreads, prices and movement.

The pipeline should therefore be separated into:

```text
NBA statistics -> team, player, lineup, schedule and result features
injury reports -> timestamped availability and expected-lineup evidence
market source   -> timestamped spreads, prices, totals and settlements
forecast model  -> margin distribution and cover probability
policy layer    -> leakage, freshness, uncertainty and risk checks
```

## Guardrails

- Never use information published after the forecast timestamp.
- Never describe one game as guaranteed, safe or certain.
- Keep model forecasts separate from market prices in stored data.
- Log every manual adjustment and its evidence.
- Prefer paper evaluation until calibration and data quality are demonstrated.
- Report negative and flat results honestly.
- Evaluate performance over a chronological held-out period.
- Do not promote a learned rule into `notebook.json` unless the gate approves
  it using only earlier data.

The primary analytical comparison is:

```text
predicted margin distribution versus the market spread and price
```

It is not merely a prediction of which team will win.

