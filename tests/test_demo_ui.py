import math

from demo_ui.adapters import analyse_scenario, inplay_story, load_runtime, scenarios
from demo_ui.lifecycle import execution_stages


def test_demo_runtime_has_offline_scenarios():
    runtime = load_runtime()
    slate = scenarios(runtime)
    assert slate
    assert runtime.model_status in {"TRAINED ARTIFACT", "DETERMINISTIC FALLBACK"}


def test_demo_scenario_uses_real_agent_and_replay_values():
    runtime = load_runtime()
    analysis = analyse_scenario(runtime, scenarios(runtime)[0].game_id)
    assert analysis["provenance"] == "HISTORICAL REPLAY"
    assert analysis["decision"]["trace"]
    assert 0 < analysis["raw_probability"] < 1
    assert 0 < analysis["market_mid"] < 1
    assert math.isfinite(analysis["model_brier"])
    assert analysis["grade"]["status"] in {"settled", "passed", "not filled"}


def test_inplay_scene_is_explicitly_synthetic():
    runtime = load_runtime()
    analysis = analyse_scenario(runtime, scenarios(runtime)[0].game_id)
    story = inplay_story(runtime, analysis)
    assert story["available"]
    assert story["provenance"] == "DEMO / SYNTHETIC"
    assert len(story["checkpoints"]) >= 2


def test_execution_view_does_not_expose_internal_trace_details():
    trace = [{"step": "trigger", "detail": "private detail"}, {"step": "deliver", "detail": "sent"}]
    stages = execution_stages(trace, "Before game", acted=True)
    assert [stage["label"] for stage in stages][:2] == ["Observe", "Forecast"]
    assert "private detail" not in str(stages)
    assert execution_stages(trace, "After game", acted=False)[5]["detail"] == "no position to grade"
