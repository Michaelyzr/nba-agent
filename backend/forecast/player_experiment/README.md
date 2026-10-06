# Player feature engineering and calibration: isolated experiment

This package adds files only. It does not modify `forecast/api.py`, the GRU,
the existing feature list, training commands, model defaults, data schemas,
news agents, or the trading pipeline. It has no dependency on draft PR #6.
The base is main commit fd113461b741f66316f15f9db081c5cab1f9c8d1.

## Purpose

Test whether role/efficiency/variability features improve player points and
minutes forecasts, and whether a separate monotone probability calibrator adds
value. Reuse existing ESPN-derived records; no new feeds or API calls are needed.
Predictions are **conditional on the player appearing**, not tradable contract
probabilities: participation, injury assumptions, and void/settlement rules
remain outside this experiment.

## Run

Use the repository's installed requirements from its root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m forecast.player_experiment.benchmark \
  --source sample --out runs/player-features-v1/experiment
python -m forecast.player_experiment.diagnostics runs/player-features-v1/experiment
python -m pytest -q
```

Choose a new output folder for another run. The benchmark refuses to overwrite
an existing nonempty folder. Models, feature tables, and row-level predictions
stay in git-ignored `runs/`. Aggregate metrics and manifests are committed under
`reports/results/player_features_v1/`. Do not copy these artifacts over
`models/m1/` or `models/m2/`.

## What is compared

A: baseline feature set corresponding to the existing player model, excluding
`teammates_out_min`, which was retrospectively inferred from the target game.

B: the same baseline plus 15 features:

- Last-20-game minutes and points averages.
- Last-10-game minutes and points standard deviations.
- Last-5 minus last-20 minutes and points trends.
- Shot attempts per minute over 5 and 10 games.
- Free-throw attempts per minute and per field-goal attempt over 10 games.
- Pooled true-shooting estimates over 10 and 20 games.
- A recent shooting estimate shrunk toward the longer history with 25 attempt-equivalents.
- Whether the latest known team matches the target team, and games in that stint.

True-shooting estimate is `sum(points) / (2 * (sum(FGA) + 0.44 * sum(FTA)))`.
Ratios use pooled totals rather than unweighted averages of game percentages.
Zero denominators remain missing; the boosting model handles missing features.
Usage, pace and advanced ratings are not downloaded or included.

Both A and B use exactly the same quantile boosting configuration: five
quantiles (0.10, 0.25, 0.50, 0.75, 0.90), points and minutes targets, 150
iterations, learning rate 0.05, 15 leaves, minimum leaf size 40, L2 1, seed 7606.
There is no random early-stopping split and no hyperparameter search.
This controlled baseline is not an exact reproduction of the published GBM
or GRU: model capacity, information availability and periods differ. Do not
compare the absolute scores below directly with published GRU numbers.

C: calibrate A; D: calibrate B. Each uses its own separate January calibration
period, after the forecaster is frozen. A positive-slope sigmoid maps raw
points-event probabilities to calibrated probabilities. All thresholds share
one mapping, so P(points >= 20) cannot fall below P(points >= 30). A nonpositive
fitted slope falls back to the identity mapping. Each player-game has equal
weight across its threshold events.

Calibration applies **only to points-event probabilities**, not the raw points
or minutes quantile intervals. Outputs label raw quantiles separately. Integer
thresholds use `>=`, consistent with continuity correction in the existing
quantile-to-probability helper. No learning of participation or settlement rules
is implied.

## Historical availability and evaluation

Player summaries are computed once after each completed game, then joined to
prediction cutoffs with an as-of join. Opponent summaries follow the same rule.
They never use the target game's minutes, starting status, scores or actual
absences. Default cutoff is 30 minutes before tip. Inference uses exactly the
same builder with explicit target rows. Feature construction excludes games
whose `final_at` is after the decision.

`final_at` in the supplied data is itself an assumed completion timestamp;
this code does not establish historical publication times. We use the existing
data, and do not call this an official-NBA-data experiment.

| Window | Initial training ends | Validation | Refit ends | Calibration | Evaluation |
| --- | --- | --- | --- | --- | --- |
| 2025 | Before Oct 2024 | Oct-Dec 2024 | Before Jan 2025 | Jan 2025 | Feb 2025 through Apr 13 Eastern |
| 2026 | Before Oct 2025 | Oct-Dec 2025 | Before Jan 2026 | Jan 2026 | Feb 2026 through Apr 12 Eastern |

All intermediate boundaries are UTC. Fit/calibration labels must have completed
before their next period starts; test and validation forecasts must be made
after the preceding boundary. This avoids using a late-evening label from the
previous period that was not yet available. Player rows from the same game
share the same boundary. Rolling historical inputs can update during evaluation
as earlier games finish; model weights and calibrators remain frozen.

The January sets contain 4,780 / 4,936 player rows. The evaluation sets contain
11,035 rows over 524 games (2025), and 10,857 rows over 505 games (2026).
Validation favors the enriched features for both targets in both windows.
Both variants are retained and refitted regardless; there is no silent promotion.

These are retrospective chronological evaluations. The 2026 period overlaps
already published repo results. The earlier window offers temporal replication,
not an independently untouched prospective test. No parameters were changed
in response to either evaluation window.

## Measured results

Pooled player-points Brier score across fixed >=10, >=15, >=20, >=25, >=30,
>=35 thresholds (lower is better):

| Variant | 2025 | 2026 |
| --- | --- | --- |
| Baseline | 0.082346 | 0.081343 |
| Baseline + calibration | 0.082155 | 0.081298 |
| Enriched features | 0.081844 | 0.080909 |
| Enriched + calibration | **0.081643** | **0.080868** |

Combined relative reduction: about **0.85%** in 2025 and **0.58%** in 2026.
These are small improvements, not a major accuracy breakthrough.

| Raw distribution metric | Baseline 2025 | Enriched 2025 | Baseline 2026 | Enriched 2026 |
| --- | --- | --- | --- | --- |
| Points median MAE | 4.7179 | 4.7034 | 4.6846 | 4.6577 |
| Points pinball loss | 1.68587 | 1.67882 | 1.66324 | 1.65572 |
| Minutes median MAE | 5.2204 | 5.1992 | 5.1658 | 5.1460 |
| Minutes pinball loss | 1.83809 | 1.82070 | 1.81288 | 1.79885 |

The combined Brier gains have positive game-cluster bootstrap intervals in both
windows: [0.000460, 0.000947] and [0.000246, 0.000710]. Calibration's *incremental*
2026 gain includes zero: [-0.000077, 0.000153]. Clustering accounts for players
and thresholds within a game, but not repeated players/teams across dates;
these intervals are conditional on the fitted experiment and do not account
for all model/data-selection uncertainty.

Post-run diagnostic slices (not used to tune/select models):
- Players with prior average minutes >=20 improve in both windows; combined
  Brier changes from 0.107917 to 0.106851 and 0.105837 to 0.105048.
- Low-minute players have no consistent calibration benefit; their 2026 combined
  Brier is slightly worse (0.042069 to 0.042101).
- Fewer than five historical appearances: combined Brier worsens in both windows.
  The API flags limited history; no unvalidated corrective rule was fitted.
- Raw 10-90% points intervals cover about 87-89%, not the nominal 80%. Probability
  calibration does not fix these raw intervals. See reliability and threshold
  CSVs for remaining calibration errors. The feature-only 2025 pooled log loss
  is slightly worse despite better Brier; calibration improves the combined result.

Fixed thresholds over all played players are not equivalent to the exchange's
actual market selection. We report player/game counts, per-threshold metrics,
reliability bins and meaningful role slices to avoid hiding that limitation.
No profit, market-beating, or superiority over the deployed GRU is claimed.

## Explicit opt-in inference

```python
import pandas as pd
from forecast.player_experiment.api import PlayerExperiment

experiment = PlayerExperiment.load(
    'runs/player-features-v1/experiment/models-2026.pkl', variant='enriched')
targets = pd.DataFrame([{
    'game_id': game_id, 'player_id': player_id, 'team_id': team_id,
    'as_of': decision_time_with_timezone,
}])
results = experiment.predict(player_games, games, targets, thresholds=[20, 25, 30])
```

Use trusted local pickle artifacts only. The API rejects cutoffs earlier than
its calibration labels, post-tip targets, invalid identities and invalid
thresholds. A target's supplied outcome fields are discarded. Outputs include:
- `raw_quantiles` for points and minutes;
- raw and calibrated probabilities **given played** for each >= threshold;
- model name, timestamp, history count, and a limited-history flag;
- empty `external_sources_applied`, because no news is processed here.

Do not connect these probabilities directly to a betting agent without the
appropriate participation and settlement handling. The existing player model
API and agent interface have not been changed.

## Computing and validation

The full experiment took about 56 seconds locally, including two temporal
windows, validation/refit models for both feature sets, calibration, scoring,
and game-cluster bootstrap comparisons. Features for 84,833 played rows took
about 0.79 seconds. Prediction batches of ~11,000 rows took 0.26-0.47 seconds.
These are local timings, not a controlled speedup over the original pipeline.
Cached snapshots and shared prepared features avoid recomputing history for
every fitted model. The models require no GPU.

The existing suite passed before changes (113 tests); the extended suite passes
120 tests. New tests cover baseline-feature equivalence, unfinished/current-game
leakage, pooled ratios, training/inference parity, monotone calibration, model
serialization, conditional output semantics and model-availability cutoffs.
The host emits existing sklearn/NumPy matmul warnings on logistic paths; output
probabilities are checked for finiteness. No warnings were globally suppressed.

Next work: validate cold-start behavior, test calibrated distribution intervals,
and add pace/usage data only after checking download coverage and NBA/ESPN ID
mapping. Keep this experiment opt-in until a separate promotion decision.

AI assistance: Codex assisted implementation, review and testing. All reported
metrics were generated by executed local code.
