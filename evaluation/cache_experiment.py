"""Run with: python -m evaluation.cache_experiment

Synthetic regression diagnostic, not an NBA performance evaluation.
Does not train models, call APIs, or alter production cache behavior.
"""
import math
from pathlib import Path

import pandas as pd

from forecast.api import Forecaster
from replay import AsOf

ROOT = Path(__file__).resolve().parent.parent


def main():
    # A deliberately small synthetic schedule crossing a same-day completion.
    # Stored scores are masked by AsOf until final_at, as in the real replay.
    tables = {
        'games': pd.DataFrame([
            {'game_id': 'prior', 'date': '2026-02-01',
             'tip_time': pd.Timestamp('2026-02-01T15:00:00Z'),
             'final_at': pd.Timestamp('2026-02-01T18:00:00Z'),
             'home_team_id': 1, 'away_team_id': 2,
             'home_team': 'HOME', 'away_team': 'AWAY', 'home_pts': 110, 'away_pts': 100},
            {'game_id': 'target', 'date': '2026-02-01',
             'tip_time': pd.Timestamp('2026-02-01T22:00:00Z'),
             'final_at': pd.Timestamp('2026-02-02T01:00:00Z'),
             'home_team_id': 1, 'away_team_id': 2,
             'home_team': 'HOME', 'away_team': 'AWAY', 'home_pts': 101, 'away_pts': 99},
        ]),
        'player_games': pd.DataFrame([
            {'game_id': 'prior', 'player_id': 7, 'team_id': 1,
             'min': 34., 'pts': 25., 'fga': 18., 'fta': 6., 'started': True},
        ]),
    }
    game = next(g for g in tables['games'].itertuples() if g.game_id == 'target')
    early = AsOf(tables, pd.Timestamp('2026-02-01T17:00:00Z'), {})
    late = AsOf(tables, pd.Timestamp('2026-02-01T19:00:00Z'), {})
    try:
        model = Forecaster.load()
    except FileNotFoundError:
        print('Local models are missing. Run: python -m forecast.train --source sample --out runs/local-training')
        return 2

    rows = []

    def capture(label, view):
        p = model.win(view, game)['p_home']
        history = model.history(view)
        rows.append({'case': label, 'decision_time_utc': view.now.isoformat(),
                     'visible_finished_games': len(view.games().dropna(subset=['home_pts'])),
                     'used_team_games': len(history.team_games(1, game.tip_time)),
                     'p_home': p})
        return p

    early_fresh = capture('17:00 directly', early)
    late_after_early = capture('19:00 after 17:00', late)
    model = Forecaster.load()
    late_fresh = capture('19:00 directly', late)
    model = Forecaster.load()
    capture('Reverse order: 19:00 first', late)
    early_after_late = capture('17:00 after 19:00', early)

    frame = pd.DataFrame(rows)
    display = frame.rename(columns={'case': 'Access sequence', 'decision_time_utc': 'Decision time (UTC)',
                                    'visible_finished_games': 'Visible completed games',
                                    'used_team_games': 'Team games used',
                                    'p_home': 'Home win probability'})
    display['Home win probability'] = display['Home win probability'].map(lambda p: f'{p:.2%}')
    print('Experiment: Does access order change the prediction at the same decision time?\n')
    print('Synthetic data: the prior game finishes at 18:00; the target game starts at 22:00. All times are UTC.')
    print('This experiment tests cache correctness, not NBA prediction performance or profitability.\n')
    print(display.to_string(index=False))
    forward_ok = math.isclose(late_after_early, late_fresh, rel_tol=0, abs_tol=1e-10)
    backward_ok = math.isclose(early_after_late, early_fresh, rel_tol=0, abs_tol=1e-10)
    print(f'\nAccess-order difference at 19:00: {abs(late_after_early-late_fresh)*100:.3f} percentage points')
    print(f'Access-order difference at 17:00: {abs(early_after_late-early_fresh)*100:.3f} percentage points')
    folder = ROOT / 'runs' / 'cache-experiment'
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / 'results.csv', index=False)
    print(f'\nResults saved to: {folder / "results.csv"}')
    if forward_ok and backward_ok:
        print('PASS: Access order did not affect predictions in this experiment.')
        return 0
    print('FAIL: The cache issue was reproduced. Predictions depend on access order; reverse access may also use future data.')
    print('FAIL indicates a reproduced correctness defect. After fixing the cache, rerunning should produce PASS.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
