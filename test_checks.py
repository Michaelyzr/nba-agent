"""Planted failures must be rejected; the loop must retry, pause, and stop as designed."""
import copy
import json
import math
import os

import pytest

os.environ.setdefault("NBA_AGENT_OFFLINE", "1")

import checks  # noqa: E402
from graph import reply, start  # noqa: E402
from tables import EVAL, MODELS, load  # noqa: E402

QUESTIONS = json.loads((EVAL / "questions.json").read_text())


def first(kind):
    return next(q for q in QUESTIONS if q["type"] == kind)


def names(failures):
    return {f["check"] for f in failures}


@pytest.fixture(scope="module")
def runs():
    return {kind: start(first(kind)["question"]) for kind in ("trade_split", "rate", "team_window", "context", "signing")}


def test_clean_runs_pass(runs):
    for kind, out in runs.items():
        assert out["status"] == "answered", (kind, out["note"])
        assert checks.check_result(out["form"], out["result"]) == []


def test_playoff_mix(runs):
    r = copy.deepcopy(runs["trade_split"]["result"])
    r["windows"][1]["game_ids"].append(load("player_playoffs").GAME_ID.iloc[0])
    r["windows"][1]["games"] += 1
    assert "playoff_mix" in names(checks.check_result(runs["trade_split"]["form"], r))


def test_wrong_team(runs):
    r = copy.deepcopy(runs["trade_split"]["result"])
    r["windows"][0]["team_id"] = r["windows"][1]["team_id"]
    assert "wrong_team" in names(checks.check_result(runs["trade_split"]["form"], r))


def test_missing_game_count(runs):
    r = copy.deepcopy(runs["rate"]["result"])
    del r["windows"][0]["games"]
    assert "no_game_count" in names(checks.check_result(runs["rate"]["form"], r))


def test_rate_basis(runs):
    r = copy.deepcopy(runs["rate"]["result"])
    r["comparison"] = {"a": "pts_per_game", "b": "pts_per_100"}
    assert "rate_basis" in names(checks.check_result(runs["rate"]["form"], r))


def test_stat_mismatch(runs):
    r = copy.deepcopy(runs["team_window"]["result"])
    r["windows"][0]["values"]["net_rating"] += 2.0
    assert "stat_mismatch" in names(checks.check_result(runs["team_window"]["form"], r))


def test_invented_quote(runs):
    r = copy.deepcopy(runs["context"]["result"])
    r["quotes"][0]["text"] += " He will be back next week."
    assert "quote_not_saved" in names(checks.check_result(runs["context"]["form"], r))


def test_paragraph_about_other_player(runs):
    r = copy.deepcopy(runs["context"]["result"])
    other = load("paragraphs").loc[lambda f: f.PLAYER_ID != r["subject"]["id"]].iloc[0]
    r["paragraphs"].append({"para_id": other.PARA_ID})
    assert "wrong_paragraph" in names(checks.check_result(runs["context"]["form"], r))


def test_signing_needs_label_and_error(runs):
    r = copy.deepcopy(runs["signing"]["result"])
    r.pop("heldout_mae")
    r["label"] = "fact"
    assert {"no_heldout_error", "unlabeled_prediction"} <= names(checks.check_result(runs["signing"]["form"], r))


def test_note_number_needs_citation(runs):
    out = runs["rate"]
    note = out["note"].replace("[W:window]", "", 1)
    assert "no_game_list" in names(checks.check_note(out["form"], out["note_result"], note))


def test_note_number_must_match(runs):
    out = runs["trade_split"]
    v = out["result"]["windows"][0]["values"]["ts_pct"]
    note = out["note"].replace(f"{v:.3f}", f"{v + 0.05:.3f}", 1)
    assert "stat_mismatch" in names(checks.check_note(out["form"], out["note_result"], note))


def test_signing_note_rules(runs):
    out = runs["signing"]
    note = out["note"].replace("prediction", "estimate") + " The team has cap room for this."
    got = names(checks.check_note(out["form"], out["note_result"], note))
    assert "cap_claim" in got


def test_context_note_cannot_adopt_article_number():
    trade_q = next(q for q in QUESTIONS if q["type"] == "context" and "trade" in q["question"])
    out = start(trade_q["question"])
    note = out["note"].replace("Saved reports about", "He will play 32 minutes a night [W:box]. Saved reports about", 1)
    assert names(checks.check_note(out["form"], out["note_result"], note)) & {"article_number", "no_game_list", "stat_mismatch"}


@pytest.mark.parametrize("plant,expected", [("playoffs", "playoff_mix"), ("wrong_team", "wrong_team")])
def test_loop_retries_after_planted_fault(plant, expected):
    out = start(first("trade_split")["question"], plant=plant)
    rejected = [t for t in out["trace"] if t["step"] == "checks.py" and expected in t["detail"]]
    assert rejected and out["status"] == "answered" and out["tries"] == 1


def test_note_retry_after_planted_number():
    out = start(first("rate")["question"], plant="note_number")
    assert out["note_tries"] == 1 and out["status"] == "answered"


def test_followup_pauses_then_resumes():
    out = start("How did he shoot after the trade?")
    assert out["paused"] and "who" in out["missing"]
    t = load("trades").iloc[0]
    name = load("players").set_index("PLAYER_ID").loc[t.PLAYER_ID, "PLAYER_NAME"]
    out = reply(f"{name}, this season", out["config"])
    assert not out["paused"] and out["status"] == "answered"


def test_out_of_scope_is_rejected():
    out = start(next(q for q in QUESTIONS if "MVP" in q["question"])["question"])
    assert out["paused"] and "metric" in out["missing"] and "does not predict" in out["followup"]


def test_heldout_contract_error_exists():
    m = json.loads((MODELS / "signing" / "metrics.json").read_text())
    assert m["n_heldout"] > 0 and math.isfinite(m["mlp_mae"]) and m["mlp_mae"] > 0
