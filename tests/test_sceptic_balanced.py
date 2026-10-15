"""Balanced scored sceptic: parsing, fail-closed behaviour, threshold, default mode unchanged (no network)."""
import pytest

from agents.llm_client import CachedLLM, StubBackend
from agents.tool_agent import (SCEPTIC_BALANCED_SYSTEM, SCEPTIC_SYSTEM, ToolAgent, heuristic_reply, parse_balanced,
                               sceptic_prompt)
from tests.test_tool_agent import _forecaster, run, tables_with_history


def reply(json=None, error=None):
    return {"json": json, "text": "", "error": error}


GOOD = {"score": 0.62, "for": "Fresh news.", "against": "Small shift.", "reason": "For outweighs against."}


def test_parse_valid_score_and_threshold():
    out = parse_balanced(reply(GOOD), threshold=0.5)
    assert out["verdict"] == "approve" and out["score"] == pytest.approx(0.62) and "invalid" not in out
    assert out["for"] == "Fresh news." and out["against"] == "Small shift."
    assert parse_balanced(reply(GOOD), threshold=0.7)["verdict"] == "reject"
    assert parse_balanced(reply({**GOOD, "score": 0.5}), threshold=0.5)["verdict"] == "approve"   # >= threshold
    assert parse_balanced(reply({**GOOD, "score": "0.8"}))["score"] == pytest.approx(0.8)         # numeric string


@pytest.mark.parametrize("msg", [None, {}, {"verdict": "approve"}, {**GOOD, "score": 1.2}, {**GOOD, "score": -0.1},
                                 {**GOOD, "score": "high"}, {**GOOD, "score": True}, {**GOOD, "score": float("nan")},
                                 {**GOOD, "score": None}, {**GOOD, "score": [0.6]}])
def test_parse_invalid_fails_closed(msg):
    out = parse_balanced(reply(msg), threshold=0.0)
    assert out["verdict"] == "reject" and out["score"] == 0.0 and out["invalid"]


def test_parse_llm_error_fails_closed():
    out = parse_balanced(reply(GOOD, error="RuntimeError: HTTP 500"), threshold=0.0)
    assert out["verdict"] == "reject" and out["score"] == 0.0 and out["invalid"]


@pytest.mark.parametrize("field", ["reason", "for", "against"])
def test_parse_banned_wording_fails_closed(field):
    out = parse_balanced(reply({**GOOD, field: "This is a guaranteed win."}), threshold=0.0)
    assert out["verdict"] == "reject" and out["score"] == 0.0 and "banned" in out["reason"]


def test_parse_truncates_text():
    out = parse_balanced(reply({**GOOD, "reason": "x" * 1000}))
    assert len(out["reason"]) == 300


def test_prompt_suffix_by_mode():
    assert sceptic_prompt("H", {"a": 1}).endswith("Reply with your JSON verdict.")
    assert sceptic_prompt("H", {"a": 1}, "balanced").endswith("Reply with your JSON score.")
    assert "score" in SCEPTIC_BALANCED_SYSTEM and "Case FOR" in SCEPTIC_BALANCED_SYSTEM


def make(fn, **kw):
    llm = CachedLLM(StubBackend(fn, model="test-stub"), cache_dir=None)
    return ToolAgent(llm, forecaster=_forecaster(), players={7: "Test Player"}, **kw)


def scored(score):
    def fn(role, system, prompt, meta):
        if role == "sceptic_balanced":
            assert system == SCEPTIC_BALANCED_SYSTEM
            return {**GOOD, "score": score} if score is not None else "no json here"
        assert role != "sceptic"
        return heuristic_reply(role, system, prompt, meta)
    return fn


def test_default_mode_is_verdict():
    seen = []

    def fn(role, system, prompt, meta):
        seen.append(role)
        if role == "sceptic":
            assert system == SCEPTIC_SYSTEM
        return heuristic_reply(role, system, prompt, meta)
    a = make(fn)
    assert a.sceptic_mode == "verdict" and a.setup == "tool_sceptic"
    run(a, tables_with_history())
    assert "sceptic" in seen and "sceptic_balanced" not in seen


def test_balanced_mode_threshold_decides():
    a = make(scored(0.62), sceptic_mode="balanced", sceptic_threshold=0.55)
    assert a.setup == "tool_sceptic_balanced"
    _, fills, traces = run(a, tables_with_history())
    assert len(fills) == 1 and traces[0]["sceptic"]["score"] == pytest.approx(0.62)
    b = make(scored(0.62), sceptic_mode="balanced", sceptic_threshold=0.7)
    _, fills, traces = run(b, tables_with_history())
    assert fills.empty and traces[0]["status"] == "sceptic_reject" and "candidate" in traces[0]


def test_balanced_mode_invalid_output_blocks_order():
    a = make(scored(None), sceptic_mode="balanced", sceptic_threshold=0.0)
    _, fills, traces = run(a, tables_with_history())
    assert fills.empty and traces[0]["status"] == "sceptic_reject"
    assert traces[0]["sceptic"]["invalid"] and traces[0]["sceptic"]["score"] == 0.0


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        make(heuristic_reply, sceptic_mode="vibes")


def streaming(body: bytes, monkeypatch):
    import httpx
    from evaluation import sceptic_balanced as sb
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=body)
    b = object.__new__(sb.StreamingDeepSeek)
    b.model, b.retries, b.max_tokens = sb.REASONER, 2, None
    b._body_of = lambda system, prompt: {"model": sb.REASONER, "messages": []}
    b.client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x")
    return b, calls


def test_stream_keepalives_only_fails_after_retries(monkeypatch):
    from evaluation import sceptic_balanced as sb
    monkeypatch.setattr(sb, "FIRST_TOKEN_S", -1.0)
    b, calls = streaming(b": keep-alive\n\n" * 5, monkeypatch)
    with pytest.raises(RuntimeError, match="no token"):
        b.generate("s", "p")
    assert len(calls) == 3


def test_stream_collects_content_and_usage(monkeypatch):
    import json
    chunks = [{"choices": [{"delta": {"reasoning_content": "hmm"}}]},
              {"choices": [{"delta": {"content": '{"score": '}}]}, {"choices": [{"delta": {"content": "0.4}"}}]},
              {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 7,
                                        "completion_tokens_details": {"reasoning_tokens": 3}}}]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    b, calls = streaming(body.encode(), monkeypatch)
    text, usage = b.generate("s", "p")
    assert text == '{"score": 0.4}' and len(calls) == 1
    assert usage == {"tokens_in": 10, "tokens_out": 7, "reasoning_tokens": 3, "reasoning_chars": 3}
    assert json.loads(calls[0].content)["stream"] is True
