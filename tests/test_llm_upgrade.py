"""Unit tests for the opt-in majority-vote sceptic and LLM probability blend helpers."""
from agents.tool_agent import blend_prob, majority, parse_prob, parse_verdict, vote_role


def test_parse_verdict_approve_and_fail_closed():
    assert parse_verdict({"json": {"verdict": "approve", "reason": "ok"}, "error": None})["verdict"] == "approve"
    bad = parse_verdict({"json": None, "error": "boom"})
    assert bad["verdict"] == "reject" and bad.get("invalid")
    banned = parse_verdict({"json": {"verdict": "approve", "reason": "this is a sure thing"}, "error": None})
    assert banned["verdict"] == "reject" and banned.get("invalid")


def test_majority_needs_strict_majority():
    a = {"verdict": "approve", "reason": "a"}
    r = {"verdict": "reject", "reason": "r"}
    m = majority([a, a, a, r, r])
    assert m["verdict"] == "approve" and m["approve_fraction"] == 0.6 and m["votes"].count("approve") == 3
    assert majority([a, a, r, r, r])["verdict"] == "reject"
    assert majority([a, a, r, r])["verdict"] == "reject"          # tie fails closed
    assert majority([])["verdict"] == "reject"


def test_vote_role_and_blend_and_parse_prob():
    assert vote_role(1) == "sceptic_vote1" and vote_role(5) == "sceptic_vote5"
    assert blend_prob(0.5, 0.7, 0.0) == 0.5
    assert abs(blend_prob(0.5, 0.7, 0.5) - 0.6) < 1e-9
    assert blend_prob(0.5, 0.7, 1.0) == 0.7
    assert blend_prob(0.01, 0.0, 1.0) == 0.01                     # clipped to P_MIN
    ok = parse_prob({"json": {"p_home": 0.62, "confidence": "medium"}, "error": None})
    assert ok == {"p_home": 0.62, "confidence": "medium"}
    assert parse_prob({"json": {"p_home": 1.5}, "error": None}) is None
    assert parse_prob({"json": None, "error": "x"}) is None
