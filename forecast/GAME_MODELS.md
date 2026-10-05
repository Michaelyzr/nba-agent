# Stats-only game forecasts: first improvement pass

The new models are opt-in. The existing agent continues to use `forecast.api.Forecaster`.
No exchange orders, market loader changes, or default artifact replacement are included.

## Reproduce

From the repository root, with `requirements.txt` installed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m forecast.benchmark_games --source sample --out runs/game-baseline-v1
python -m pytest -q
```

The committed sample contains 3,962 games and 85,422 player box-score rows, not
just a few example games. It uses ESPN-derived records. This experiment uses
scores and schedules, not market prices, synthetic data, or reconstructed
inactive-list timestamps. The sample's `final_at` is an assumed completion time;
these results do not establish historical availability beyond that assumption.

Outputs are `validation.csv`, `metrics.csv`, `predictions.csv`, `features.parquet`,
`manifest.json`, and `models.pkl`. Features are constructed once for every
candidate. Raw features, game-level predictions and trained artifacts remain
under git-ignored `runs/`; aggregate results are in `evaluation/results/game_v1/`.
The manifest records the input SHA-256, partitions, artifact availability time,
and runtime. Load pickle bundles only from trusted local training runs.

## What changed

- New stats-only team features: prior scoring/defense at 5/10/30-game windows,
  form, variability, rest, home-away differences, and sequential Elo ratings.
  Only games with `final_at <= as_of` enter history. Default cutoff: tip minus 30 minutes.
- A richer regularized logistic win classifier, compared with the existing
  four-feature stats-only logistic architecture retrained without absence inputs.
- Margin and total center models (rolling, ridge, shallow histogram boosting),
  with empirical residual distributions estimated on a separate January period.
  Negative margins are retained; probabilities distinguish `>` from `>=`.
  Margin probabilities exclude tied final outcomes; home/away win probabilities sum to one.
- Opt-in market adapter for winner/spread/total. Winner and spread share the
  margin distribution. The richer classifier probability is explicitly a comparison field.
- Optional adapter to the existing player model, with separate YES/NO/VOID
  probabilities and explicit non-participation rules. This is not a player-model
  improvement claim or a complete exchange settlement implementation.
- Existing GRU validation now splits at a whole timestamp/game boundary and fits
  normalizers on training rows only. It rejects insufficient chronological history.
- Existing forecast history cache is instance-local and keyed by exact cutoff,
  rather than date, to avoid stale same-day histories and source-ID reuse.

## Experiment protocol

| Partition | UTC boundary / count |
| --- | --- |
| Initial training | Before 2025-10-01; 2,640 games |
| Selection validation | 2025-10-01 through 2025-12-31; 492 games |
| Refit after selection | Before 2026-01-01; 3,132 games |
| Residual calibration | January 2026; 233 games |
| Regression benchmark | From 2026-02-01 UTC through April 12 Eastern; 506 games |

Win candidates: C = 0.01, 0.1, 1, 10, selected by validation Brier.
Score candidates: rolling; ridge alpha = 10, 100, 1000; shallow boosting = 100,
250 iterations, selected by validation MAE. No random early-stopping split.
Selected: C = 0.01 and ridge alpha = 1000 for both score targets.

The test period overlaps already published repository results. This is a
regression benchmark, not newly untouched evidence. The ending boundary was
corrected from UTC midnight to Eastern midnight before the final report; no
candidate or hyperparameter was changed in response to test performance.
Unlike the existing market report's 501 games, this cohort does not require
market-price availability. All compared predictions use the same 506 games.

## Results

| Forecast / metric (lower is better) | Baseline | Candidate |
| --- | --- | --- |
| Winner Brier, stats-only logistic | 0.184720 | 0.182147 |
| Winner log loss, stats-only logistic | 0.551993 | 0.546853 |
| Margin MAE, versus rolling | 12.3856 | 11.8054 |
| Margin pinball loss, versus rolling | 4.35802 | 4.17796 |
| Total MAE, versus rolling | 15.5680 | 15.5629 |
| Total pinball loss, versus rolling | 5.44869 | 5.45620 |

Winner accuracy rises from 73.32% to 75.10%. Brier improvement is 0.002573;
95% paired-game bootstrap interval is [-0.000376, 0.005551]. The interval
includes zero, and the bootstrap does not model cross-game temporal dependence.
Do not claim a conclusive win-model improvement.

The margin-derived winner has Brier 0.185191; the richer logistic classifier
has Brier 0.182147. They are separate outputs, not interchangeable evidence.
Totals are effectively flat and slightly worse on distribution loss; no total
probability improvement is established. All five fixed spread thresholds improve
on their rolling benchmark. Thresholds are diagnostic lines, not historical
market lines or proof of profitable trading.

The legacy as-coded diagnostic scores 0.177289 Brier, numerically better than
the new models. It uses retrospective actual-absence features and trains through
January, whereas the clean comparison reserves January for calibration. It is
reported for transparency, not used as an apples-to-apples promotion target.
This branch does not claim to outperform every existing repository result.

Feature construction took about 0.32 seconds and the complete game experiment
about 7.3 seconds on the development host (Python 3.13, NumPy 2.2.6, scikit-learn
1.7.0). This is a measured runtime, not a controlled speedup against the old
training command, which trains different models. The host emitted numerical
matmul warnings in both existing and new sklearn paths; reported predictions
were finite and the tests passed. Retest timings and numerics on the deployment
runtime before promotion.

## Use the opt-in API

```python
from forecast.market_forecaster import MarketForecaster

f = MarketForecaster.load("runs/game-baseline-v1/models.pkl")
# `view` is replay.AsOf; `game` is a game row from view.games().itertuples().
winner = f.predict(view, game, "winner", side="home")
spread = f.predict(view, game, "spread", threshold=4.5, side="home")
total = f.predict(view, game, "total", threshold=225.5)
```

`probability` answers the specified event. Winner is the margin > 0 event.
The winner response also exposes `comparison_logit_probability`, which is not
silently substituted into the margin model. Margin quantiles are explicitly
home-team margin diagnostics even when the selected outcome is the away side.
The adapter rejects cutoffs earlier than its calibration outcomes, post-tip
forecasts, regulation-only requests, and unimplemented news overrides.
It returns `unsupported` for championship simulation.

To reuse a locally trained existing player model:

```python
from forecast.api import Forecaster
f = MarketForecaster.load("runs/game-baseline-v1/models.pkl", Forecaster.load())
player = f.predict(view, game, "player_points", threshold=30, operator=">=",
                   player_id=player_id, participation_rule="void_if_not_play")
```

Player output separates `p_yes_if_play`, `p_play`, `p_yes`, `p_no`, `p_void`.
Do not feed `p_yes` alone to an agent that assumes only two settlement states.
This adapter does not verify a venue's contract text or repair the legacy
player model's retrospective absence training. Existing player models and their
historical training cutoffs must be checked before using them in a replay.

## Limits and next work

- Empirical residual distributions are global and assume constant uncertainty.
  They are not jointly modeled score distributions. Quantile diagnostics are
  continuous approximations; event probabilities apply integer boundaries.
- Default Elo parameters and cold-start 110-point priors are fixed assumptions.
  No season reset, roster-change strength correction, or pace adjustment yet.
- The full agent remains game-winner-only. Market ingestion, side/line mapping,
  participation/void settlement, and downstream risk checks must be extended
  before trading other market types.
- Championship simulation, richer player probability calibration, and retraining
  the player models without hindsight are not implemented in this first pass.
- New forecasts remain opt-in. Future promotion needs temporal replication and
  comparison on actual market lines; a larger NN is not justified by these results.

AI assistance: implemented and reviewed with Codex; results are copied from
local execution, not generated estimates.
