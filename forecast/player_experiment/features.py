"""Completed-game snapshots joined to pregame targets. No current-game availability.

The sample's final_at is a source assumption, not a verified publication time.
Targets are conditional on participation; this builder does not predict rosters.
"""
import numpy as np
import pandas as pd

from forecast.features import FEATURES as LEGACY_FEATURES

BASE_FEATURES = tuple(f for f in LEGACY_FEATURES if f != 'teammates_out_min')
EXTRA_FEATURES = (
    'min_avg20', 'pts_avg20', 'min_std10', 'pts_std10', 'minutes_trend',
    'points_trend', 'shots_per_min5', 'shots_per_min10', 'fta_per_min10',
    'free_throw_rate10', 'ts10', 'ts20', 'ts10_shrunk',
    'same_team_as_last', 'current_stint_games',
)
ENRICHED_FEATURES = BASE_FEATURES + EXTRA_FEATURES


def _ratio(numerator, denominator):
    return numerator / denominator.where(denominator > 0)


def _snapshots(played):
    """Compute each player's state once after each completed game."""
    h = played.sort_values(['player_id', 'final_at', 'game_id']).reset_index(drop=True)
    group = h.groupby('player_id', sort=False)
    for field in ('min', 'pts', 'fga', 'fta', 'started'):
        for window in (5, 10, 20):
            h[f'{field}_avg{window}'] = group[field].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
    for field in ('min', 'pts'):
        h[f'{field}_std10'] = group[field].transform(
            lambda x: x.rolling(10, min_periods=2).std(ddof=0))
    for window in (10, 20):
        sums = {field: group[field].transform(lambda x: x.rolling(window, min_periods=1).sum())
                for field in ('pts', 'fga', 'fta')}
        equivalent_attempts = sums['fga'] + .44 * sums['fta']
        h[f'ts{window}'] = _ratio(sums['pts'], 2 * equivalent_attempts)
        if window == 10:
            h['attempt_equivalents10'] = equivalent_attempts
    # Shrink noisy short-window efficiency toward the player's longer history.
    h['ts10_shrunk'] = _ratio(h.ts10 * h.attempt_equivalents10 + 25 * h.ts20,
                             h.attempt_equivalents10 + 25)
    h['pts_per_min_avg10'] = _ratio(h.pts_avg10, h.min_avg10)
    for w in (5, 10):
        h[f'shots_per_min{w}'] = _ratio(h[f'fga_avg{w}'], h[f'min_avg{w}'])
    h['fta_per_min10'] = _ratio(h.fta_avg10, h.min_avg10)
    h['free_throw_rate10'] = _ratio(h.fta_avg10, h.fga_avg10)
    h['minutes_trend'] = h.min_avg5 - h.min_avg20
    h['points_trend'] = h.pts_avg5 - h.pts_avg20
    h['games_prior'] = group.cumcount() + 1
    changed = h.team_id.ne(group.team_id.shift())
    stint = changed.groupby(h.player_id).cumsum()
    h['current_stint_games'] = h.groupby([h.player_id, stint]).cumcount() + 1
    h['last_team_id'] = h.team_id
    h['last_tip'] = h.tip_time
    fields = list(dict.fromkeys(
        [f for f in ENRICHED_FEATURES if f in h] + ['last_team_id', 'last_tip']))
    return h[['player_id', 'final_at'] + fields]


def build_rows(player_games, games, targets=None, lead_minutes=30):
    """One row per played game, or explicitly supplied (game, player, team) target.

    With targets, outputs do not retain any target outcome columns. Historical
    scores may be present in games; as-of joins still exclude unavailable results.
    """
    g = games.copy()
    for field in ('tip_time', 'final_at'):
        g[field] = pd.to_datetime(g[field], utc=True)
    if g.game_id.duplicated().any() or g[['tip_time', 'final_at']].isna().any().any():
        raise ValueError('Unique games and complete availability timestamps required')
    if (g.final_at < g.tip_time).any():
        raise ValueError('final_at precedes tip_time')
    if player_games.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Duplicate player/game rows')
    p = player_games[player_games['min'] > 0].copy()
    p['started'] = pd.to_numeric(p.started.astype(object), errors='coerce')
    p = p.merge(g[['game_id', 'tip_time', 'final_at']], on='game_id', validate='many_to_one')
    if p.empty:
        raise ValueError('At least one historical played-game row is required')
    history = _snapshots(p)
    if targets is None:
        t = p[['game_id', 'player_id', 'team_id', 'pts', 'min']].copy()
    else:
        fields = ['game_id', 'player_id', 'team_id'] + (['as_of'] if 'as_of' in targets else [])
        t = targets[fields].copy().assign(pts=np.nan, min=np.nan)
    if t.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Duplicate target player/game rows')
    t = t.merge(g[['game_id', 'tip_time', 'final_at', 'home_team_id', 'away_team_id']],
                on='game_id', how='left', validate='many_to_one')
    t['as_of'] = (pd.to_datetime(t.as_of, utc=True) if 'as_of' in t
                  else t.tip_time - pd.Timedelta(minutes=lead_minutes))
    if t.tip_time.isna().any() or t.as_of.isna().any() or (t.as_of >= t.tip_time).any():
        raise ValueError('Targets require a known game and a pre-tip cutoff')
    if not ((t.team_id == t.home_team_id) | (t.team_id == t.away_team_id)).all():
        raise ValueError('Target team must participate in target game')
    rows = pd.merge_asof(t.sort_values('as_of'), history.sort_values('final_at'),
                         left_on='as_of', right_on='final_at', by='player_id',
                         suffixes=('', '_history'), direction='backward')
    rows['home'] = (rows.team_id == rows.home_team_id).astype(int)
    rows['opp_team_id'] = np.where(rows.home == 1, rows.away_team_id, rows.home_team_id)
    rows['rest_days'] = ((rows.tip_time - rows.last_tip).dt.total_seconds() / 86400).clip(upper=10)
    rows['back_to_back'] = (rows.rest_days < 1.5).astype(int)
    rows['games_prior'] = rows.games_prior.fillna(0)
    rows['same_team_as_last'] = (rows.team_id == rows.last_team_id).astype(int)
    rows['current_stint_games'] = rows.current_stint_games.where(rows.same_team_as_last == 1, 0).fillna(0)
    # Opponent defense uses only completed games, including for future inference.
    done = g.dropna(subset=['home_pts', 'away_pts'])
    sides = pd.concat([
        pd.DataFrame({'opp_team_id': done.home_team_id, 'final_at': done.final_at, 'allowed': done.away_pts}),
        pd.DataFrame({'opp_team_id': done.away_team_id, 'final_at': done.final_at, 'allowed': done.home_pts})
    ]).sort_values(['opp_team_id', 'final_at'])
    sides['opp_allowed_avg10'] = sides.groupby('opp_team_id').allowed.transform(
        lambda x: x.rolling(10, min_periods=1).mean())
    rows = pd.merge_asof(rows.sort_values('as_of'),
                         sides[['opp_team_id', 'final_at', 'opp_allowed_avg10']].sort_values('final_at'),
                         left_on='as_of', right_on='final_at', by='opp_team_id',
                         suffixes=('', '_opponent'), direction='backward')
    metadata = ['game_id', 'player_id', 'team_id', 'tip_time', 'as_of', 'final_at', 'pts', 'min']
    return rows[metadata + list(ENRICHED_FEATURES)].sort_values(
        ['tip_time', 'game_id', 'player_id']).reset_index(drop=True)
