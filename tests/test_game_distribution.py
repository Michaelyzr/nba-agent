import numpy as np
import pandas as pd
import pytest

from forecast.game_distribution import FEATURES, ScoreDistribution, game_rows
from forecast.market_forecaster import MarketForecaster
from test_forecast_models import league, view


def test_game_features_cannot_see_current_or_unfinished_results():
    pg, games=league()
    original=game_rows(games)
    changed=games.copy()
    changed.loc[changed.index>=20,'home_pts']=999
    after=game_rows(changed)
    pd.testing.assert_frame_equal(original.iloc[:21][FEATURES],after.iloc[:21][FEATURES])
    # A game that started earlier but has not finished must not enter history.
    games.loc[19,'final_at']=games.loc[20,'tip_time']+pd.Timedelta(hours=1)
    before=game_rows(games)
    games.loc[19,'home_pts']=999
    after=game_rows(games)
    pd.testing.assert_series_equal(before.iloc[20][FEATURES],after.iloc[20][FEATURES])


def test_target_features_match_batch_and_do_not_need_labels():
    _,games=league()
    batch=game_rows(games)
    target=games.iloc[[30]].drop(columns=['home_pts','away_pts'])
    single=game_rows(games,targets=target)
    np.testing.assert_allclose(single[FEATURES],batch.iloc[[30]][FEATURES])
    assert single.margin.isna().all()
    with pytest.raises(ValueError,match='before tip'):
        game_rows(games,targets=target.assign(as_of=target.tip_time))


def test_distribution_preserves_negative_margins_and_threshold_semantics():
    _,games=league(n_games=100)
    rows=game_rows(games)
    model=ScoreDistribution('margin').fit(rows.iloc[:50]).calibrate(rows.iloc[50:80])
    x=rows.iloc[80:]
    probs=np.array([model.probability(x,t) for t in [-10.5,-4.5,0,4.5,10.5]])
    assert np.isfinite(probs).all() and (np.diff(probs,axis=0)<=0).all()
    np.testing.assert_allclose(model.probability(x,0)+model.probability(x,0,side='away'),1.)
    np.testing.assert_allclose(model.probability(x,4.5),model.probability(x,5,operator='>='))
    assert (model.quantiles(x)<0).any()
    assert np.all(np.diff(model.quantiles(x),axis=1)>=0)
    with pytest.raises(ValueError): model.probability(x,np.nan)
    with pytest.raises(ValueError): ScoreDistribution('margin').fit(rows).calibrate(rows.head(2))


def test_market_adapter_is_consistent_and_rejects_unsupported_context():
    pg,games=league(n_games=100)
    rows=game_rows(games)
    class Win:
        def predict_proba(self,x): return np.tile([.4,.6],(len(x),1))
    bundle={'metadata':{'available_at':games.final_at.iloc[79].isoformat()},'win':Win(),**{t+'_selected':ScoreDistribution(t).fit(rows[:50]).calibrate(rows[50:80])
                         for t in ['margin','total']}}
    f=MarketForecaster(bundle)
    game=next(games.iloc[[90]].itertuples())
    v=view(pg,games,game.tip_time-pd.Timedelta(minutes=30))
    winner=f.predict(v,game,'winner')
    spread=f.predict(v,game,'spread',0.)
    assert winner['probability']==spread['probability']
    assert f.predict(v,game,'championship')['status']=='unsupported'
    with pytest.raises(ValueError,match='calibration outcomes'):
        f.predict(view(pg,games,games.tip_time.iloc[10]),game,'winner')
    assert f.predict(v,game,'winner',out=[10])['status']=='unsupported'
    assert f.predict(v,game,'total',225.5,includes_overtime=False)['status']=='unsupported'


def test_history_cache_refreshes_when_game_finishes_same_day():
    from forecast.api import Forecaster
    pg,games=league(n_games=3)
    game=games.iloc[0]
    tables={'games':games,'player_games':pg}
    from replay import AsOf
    f=Forecaster()
    early=AsOf(tables,game.tip_time+pd.Timedelta(hours=1),{})
    late=AsOf(tables,game.tip_time+pd.Timedelta(hours=4),{})
    assert len(f.history(early).team_games(1,late.now))==0
    assert len(f.history(late).team_games(1,late.now))==1
    assert f.history(late) is f.history(late)


def test_gru_scaling_excludes_validation_rows_and_whole_games_stay_together():
    from forecast.gru import GRUForecaster
    from forecast.features import make_features
    from forecast.history import History
    pg,games=league(n_games=21)
    rows=make_features(pg,games)
    cutoff=rows.tip_time.drop_duplicates().sort_values().iloc[18]
    rows.loc[rows.tip_time>=cutoff,'pts_avg5']=1e9
    m=GRUForecaster(epochs=1,batch=128,hidden=8).fit(rows,History(pg,games),verbose=False)
    assert m.validation_start==cutoff
    assert m.st_mean['pts_avg5']<100


def test_player_adapter_preserves_at_least_and_void_semantics():
    pg,games=league(n_games=100)
    class Player:
        def player(self, view, game, pid, target, lines, out):
            self.line=lines[0]
            return {'model':'test','p_play':.8,'quantiles':{'p10':10,'p25':15,'p50':20,'p75':25,'p90':30}}
    player=Player()
    f=MarketForecaster({'metadata':{'available_at':games.final_at.iloc[79].isoformat()}},player)
    game=next(games.iloc[[90]].itertuples())
    v=view(pg,games,game.tip_time-pd.Timedelta(minutes=30))
    result=f.predict(v,game,'player_points',30,operator='>=',player_id=10,participation_rule='void_if_not_play')
    assert player.line==29.5
    assert result['p_void']==pytest.approx(.2)
    assert result['p_yes']+result['p_no']+result['p_void']==pytest.approx(1)
    over=f.predict(v,game,'player_points',30,player_id=10,participation_rule='void_if_not_play')
    assert player.line==30.5 and over['p_yes']<=result['p_yes']
    out=f.predict(v,game,'player_points',30,player_id=10,participation_rule='void_if_not_play',out=[10])
    assert out['p_void']==1 and out['p_yes']==0
    assert f.predict(v,game,'player_points',30,player_id=10)['status']=='unsupported'
