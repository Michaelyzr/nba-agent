import numpy as np
import pandas as pd
import pytest

from forecast.player_experiment.features import BASE_FEATURES, ENRICHED_FEATURES, build_rows
from forecast.player_experiment.model import PlayerBoost, PointsCalibrator, points_probabilities
from test_forecast_models import league


def test_features_match_legacy_history_without_hindsight():
    from forecast.features import make_features
    pg,games=league(n_games=30)
    old=make_features(pg,games).sort_values(['game_id','player_id']).reset_index(drop=True)
    new=build_rows(pg,games).sort_values(['game_id','player_id']).reset_index(drop=True)
    assert 'teammates_out_min' not in new
    np.testing.assert_allclose(old[list(BASE_FEATURES)],new[list(BASE_FEATURES)],equal_nan=True)


def test_current_and_unfinished_outcomes_cannot_enter_features():
    pg,games=league(n_games=30)
    before=build_rows(pg,games)
    altered=pg.copy()
    altered.loc[altered.game_id>='g010',['pts','fga','fta']]=999
    after=build_rows(altered,games)
    early=before.game_id<='g010'
    pd.testing.assert_frame_equal(before.loc[early,list(ENRICHED_FEATURES)],after.loc[early,list(ENRICHED_FEATURES)])
    # Delayed source availability excludes an older game from both player/opponent histories.
    games.loc[9,'final_at']=games.loc[10,'tip_time']+pd.Timedelta(hours=1)
    before=build_rows(pg,games)
    pg.loc[pg.game_id=='g009',['pts','fga']]=999
    games.loc[9,'away_pts']=999
    after=build_rows(pg,games)
    mask=before.game_id=='g010'
    pd.testing.assert_frame_equal(before.loc[mask,list(ENRICHED_FEATURES)],after.loc[mask,list(ENRICHED_FEATURES)])


def test_efficiency_uses_pooled_totals_and_inference_matches_training():
    pg,games=league(n_games=20)
    rows=build_rows(pg,games)
    r=rows[(rows.game_id=='g010')&(rows.player_id==10)].iloc[0]
    h=pg[(pg.game_id<'g010')&(pg.player_id==10)]
    assert r.ts10==pytest.approx(h.pts.sum()/(2*(h.fga.sum()+.44*h.fta.sum())))
    assert r.shots_per_min10==pytest.approx(h.fga.sum()/h['min'].sum())
    target=pd.DataFrame([{'game_id':'g010','player_id':10,'team_id':1,'pts':999}])
    infer=build_rows(pg,games,targets=target).iloc[0]
    np.testing.assert_allclose(r[list(ENRICHED_FEATURES)].astype(float),infer[list(ENRICHED_FEATURES)].astype(float),equal_nan=True)
    assert np.isnan(infer.pts)
    rookie=build_rows(pg,games,targets=target.assign(player_id=999)).iloc[0]
    assert rookie.games_prior==0 and np.isnan(rookie.pts_avg10)


def test_calibration_is_monotone_and_learns_overconfidence():
    # Each bin has a known empirical rate closer to .5 than its prediction.
    probabilities=np.repeat(np.array([.1,.3,.5,.7,.9]),100)[:,None]
    outcomes=np.concatenate([np.r_[np.ones(k),np.zeros(100-k)] for k in [25,40,50,60,75]])[:,None]
    cal=PointsCalibrator().fit(probabilities,outcomes)
    p=cal.predict(np.linspace(0,1,100))
    assert np.isfinite(p).all() and (np.diff(p)>=0).all()
    assert 0<cal.slope<1
    assert cal.predict(np.array([.9]))[0]<.9
    with pytest.raises(ValueError): cal.fit(probabilities,np.ones_like(outcomes))


def test_fixed_boosting_and_calibrator_round_trip(tmp_path):
    import pickle
    pg,games=league(n_games=50)
    rows=build_rows(pg,games)
    model=PlayerBoost(ENRICHED_FEATURES,iterations=5).fit(rows[rows.game_id<'g025'])
    calrows=rows[(rows.game_id>='g025')&(rows.game_id<'g040')]
    pred=model.predict(calrows)
    thresholds=np.array([10,20,30])
    p=points_probabilities(pred,thresholds)
    y=(calrows.pts.to_numpy()[:,None]>=thresholds).astype(int)
    calibrator=PointsCalibrator().fit(p,y)
    assert (np.diff(calibrator.predict(p),axis=1)<=0).all()
    for target in ['pts','min']:
        q=pred[[c for c in pred if c.startswith(target+'_q')]].to_numpy()
        assert np.isfinite(q).all() and (q>=0).all() and (np.diff(q,axis=1)>=0).all()
    restored=pickle.loads(pickle.dumps((model,calibrator)))
    np.testing.assert_allclose(restored[0].predict(calrows),pred)
    np.testing.assert_allclose(restored[1].predict(p),calibrator.predict(p))


def test_feature_builder_rejects_bad_identity_or_cutoff():
    pg,games=league(n_games=5)
    with pytest.raises(ValueError,match='Duplicate'):
        build_rows(pd.concat([pg,pg.iloc[[0]]]),games)
    target=pd.DataFrame([{'game_id':'g002','player_id':10,'team_id':1,'as_of':games.tip_time.iloc[2]}])
    with pytest.raises(ValueError,match='pre-tip'):
        build_rows(pg,games,target)


def test_experiment_api_requires_available_model_and_labels_conditional_output():
    from forecast.player_experiment.api import PlayerExperiment
    pg,games=league(n_games=50)
    rows=build_rows(pg,games)
    model=PlayerBoost(BASE_FEATURES,iterations=3).fit(rows[rows.game_id<'g025'])
    calibration=rows[(rows.game_id>='g025')&(rows.game_id<'g040')]
    thresholds=np.array([10,20,30])
    raw=points_probabilities(model.predict(calibration),thresholds)
    cal=PointsCalibrator().fit(raw,(calibration.pts.to_numpy()[:,None]>=thresholds).astype(int))
    f=PlayerExperiment({'model':model,'calibrator':cal,'available_at':calibration.final_at.max().isoformat()})
    target=pd.DataFrame([{'game_id':'g045','player_id':11,'team_id':1,
                          'as_of':games.tip_time.iloc[45]-pd.Timedelta(minutes=30)}])
    result=f.predict(pg,games,target,thresholds=[0,20,30])[0]
    assert result['conditional_on']=='played'
    assert result['events'][0]['calibrated_probability_given_play']==1
    assert result['events'][1]['calibrated_probability_given_play']>=result['events'][2]['calibrated_probability_given_play']
    with pytest.raises(ValueError,match='unavailable'):
        f.predict(pg,games,target.assign(as_of=games.tip_time.iloc[10]))
