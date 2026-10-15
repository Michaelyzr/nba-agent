"""DeepSeek backend after the move to deepseek-flash / deepseek-v4-pro (no network: mocked httpx transport)."""
import calendar
import hashlib
import json

import httpx
import pytest

from agents import llm_client
from agents.llm_client import CachedLLM, DeepSeekBackend, StubBackend, cost_usd, is_peak, resolve_deepseek

DONE = b"data: [DONE]\n\n"


def sse(*chunks) -> bytes:
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + DONE


def answer(text='{"verdict": "approve"}', tokens_in=10, tokens_out=4, reasoning=None, finish="stop"):
    chunks = []
    if reasoning:
        chunks.append({"choices": [{"delta": {"reasoning_content": reasoning}}]})
    chunks.append({"choices": [{"delta": {"content": text}, "finish_reason": finish}]})
    chunks.append({"choices": [], "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out}})
    return httpx.Response(200, content=sse(*chunks))


def backend(handler, **kw) -> DeepSeekBackend:
    client = httpx.Client(base_url="https://api.deepseek.com", transport=httpx.MockTransport(handler))
    return DeepSeekBackend(client=client, **kw)


def recording(replies):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        r = replies.pop(0)
        return r() if callable(r) else r
    return handler, bodies


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm_client.random, "uniform", lambda a, b: 1.0)
    return sleeps


# ---------------- model names and the thinking switch ----------------

def test_legacy_names_map_to_flash_with_thinking_off_and_on():
    assert resolve_deepseek("deepseek-chat") == ("deepseek-flash", False)
    assert resolve_deepseek("deepseek-reasoner") == ("deepseek-flash", True)
    assert resolve_deepseek("deepseek-flash") == ("deepseek-flash", False)
    assert resolve_deepseek("deepseek-v4-pro") == ("deepseek-v4-pro", False)
    assert resolve_deepseek("deepseek-flash", thinking=True) == ("deepseek-flash", True)
    assert resolve_deepseek("deepseek-reasoner", thinking=False) == ("deepseek-flash", False)
    assert llm_client.DEEPSEEK_MODEL == "deepseek-flash" or llm_client.os.getenv("DEEPSEEK_MODEL")


def test_thinking_flag_is_sent_explicitly(no_sleep):
    handler, bodies = recording([answer(), answer(), answer()])
    backend(handler, model="deepseek-flash").generate("sys", "p")
    backend(handler, model="deepseek-flash", thinking=True).generate("sys", "p")
    backend(handler, model="deepseek-v4-pro", thinking=True, max_tokens=1000).generate("sys", "p")
    off, on, pro = bodies
    assert off["thinking"] == {"type": "disabled"} and off["temperature"] == 0.0 and "max_tokens" not in off
    assert on["thinking"] == {"type": "enabled"} and "temperature" not in on
    assert on["max_tokens"] == llm_client.REASONER_MAX_TOKENS == 32768
    assert pro["model"] == "deepseek-v4-pro" and pro["thinking"] == {"type": "enabled"} and pro["max_tokens"] == 1000
    assert all(b["stream"] and b["stream_options"] == {"include_usage": True} for b in bodies)
    assert all(b["response_format"] == {"type": "json_object"} for b in bodies)


# ---------------- truncation ----------------

def test_empty_answer_at_max_tokens_is_a_truncation_error_and_not_cached(no_sleep, tmp_path):
    handler, _ = recording([answer(text="", tokens_out=500, reasoning="hmm " * 50, finish="length")])
    llm = CachedLLM(backend(handler, model="deepseek-flash", thinking=True, max_tokens=500), cache_dir=tmp_path)
    out = llm.ask("sceptic", "sys", "p")
    assert out["truncated"] and out["error"].startswith("TruncatedError") and out["json"] is None
    assert not list(tmp_path.rglob("*.json")) and not no_sleep          # not retried, not cached


def test_empty_answer_below_the_budget_is_not_flagged_truncated(no_sleep, tmp_path):
    handler, _ = recording([answer(text="", tokens_out=20, reasoning="short")])
    llm = CachedLLM(backend(handler, model="deepseek-flash", thinking=True, max_tokens=500), cache_dir=tmp_path)
    out = llm.ask("sceptic", "sys", "p")
    assert not out["truncated"] and out["error"] is None and out["json"] is None


def test_successful_reply_has_truncated_false(no_sleep, tmp_path):
    handler, _ = recording([answer()])
    out = CachedLLM(backend(handler, model="deepseek-flash"), cache_dir=tmp_path).ask("sceptic", "sys", "p")
    assert out["truncated"] is False and out["json"] == {"verdict": "approve"}


# ---------------- stalls and backoff ----------------

class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def keepalive_stream(clock, step=20.0, lines=10):
    def gen():
        for _ in range(lines):
            clock.t += step
            yield b": keep-alive\n\n"
    return lambda: httpx.Response(200, content=gen())


def test_keepalives_without_tokens_fail_the_attempt_after_the_first_token_cap(no_sleep):
    clock = FakeClock()
    handler, bodies = recording([keepalive_stream(clock) for _ in range(6)])
    b = backend(handler, model="deepseek-flash", clock=clock)
    with pytest.raises(RuntimeError, match=r"stalled after \d+ s \(no token\)"):
        b.generate("sys", "p")
    assert len(bodies) == 1 + llm_client.DEEPSEEK_RETRIES == 6
    assert no_sleep == [2.0, 4.0, 8.0, 16.0, 32.0]                        # exponential, 2 - 60 s


def test_stall_then_recovery_returns_the_answer(no_sleep):
    clock = FakeClock()
    handler, bodies = recording([keepalive_stream(clock), answer()])
    text, usage = backend(handler, model="deepseek-flash", clock=clock).generate("sys", "p")
    assert json.loads(text) == {"verdict": "approve"} and len(bodies) == 2 and no_sleep == [2.0]


def test_slow_stream_hits_the_hard_attempt_cap(no_sleep):
    clock = FakeClock()

    def slow():
        def gen():
            for _ in range(20):
                clock.t += 15.0
                yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': 'x'}}]})}\n\n".encode()
        return httpx.Response(200, content=gen())
    handler, bodies = recording([slow, slow])
    with pytest.raises(RuntimeError, match=r"stalled after \d+ s \(slow\)"):
        backend(handler, model="deepseek-flash", thinking=True, clock=clock, max_retries=1).generate("sys", "p")
    assert len(bodies) == 2


def test_backoff_is_jittered_and_capped(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm_client.random, "uniform", lambda a, b: b)     # top of the jitter range
    handler, _ = recording([httpx.Response(503, text="busy") for _ in range(9)])
    with pytest.raises(RuntimeError, match="HTTP 503"):
        backend(handler, model="deepseek-flash", max_retries=8).generate("sys", "p")
    assert sleeps == [3.0, 6.0, 12.0, 24.0, 48.0, 60.0, 60.0, 60.0]


def test_non_retryable_status_fails_at_once(no_sleep):
    handler, bodies = recording([httpx.Response(400, text="bad model")])
    with pytest.raises(RuntimeError, match="HTTP 400"):
        backend(handler, model="no-such-model").generate("sys", "p")
    assert len(bodies) == 1 and not no_sleep


# ---------------- cache keys ----------------

def old_key(model, role, system, prompt, temperature=0.0) -> str:
    """CachedLLM.key before the migration (the published runs' cache entries)."""
    blob = json.dumps({"model": model, "temperature": temperature, "role": role, "system": system,
                       "prompt": prompt}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def test_cache_keys_are_stable():
    stub = CachedLLM(StubBackend(lambda *a: {}, model="deepseek-reasoner"), cache_dir=None)
    assert stub.key("sceptic", "s", "p") == old_key("deepseek-reasoner", "sceptic", "s", "p")
    assert stub.key("sceptic", "s", "p") == "95a21b831d2011cd4571ae6ccbc4d52e7c132e124dfdcfdf4503458af6d943f4"
    flash_on = CachedLLM(backend(lambda r: answer(), model="deepseek-flash", thinking=True), cache_dir=None)
    assert flash_on.key("sceptic", "s", "p") == "64fbdb097cec8e38189decb4fc7b2d4d29fb149682256e6aaa296f3af11c35b3"


def test_legacy_names_hit_pre_migration_entries_without_api_calls(tmp_path):
    entry = {"model": "deepseek-reasoner", "temperature": 0.0, "role": "sceptic", "system": "s", "prompt": "p",
             "text": '{"verdict": "reject"}', "tokens_in": 100, "tokens_out": 50, "reasoning_tokens": 40,
             "reasoning_chars": 120, "latency": 3.0, "created": 1760000000.0}
    k = old_key("deepseek-reasoner", "sceptic", "s", "p")
    (tmp_path / k[:2]).mkdir()
    (tmp_path / k[:2] / f"{k}.json").write_text(json.dumps(entry))

    def no_network(request):
        raise AssertionError("API called on a cached legacy request")
    reasoner = CachedLLM(backend(no_network, model="deepseek-reasoner"), cache_dir=tmp_path)
    out = reasoner.ask("sceptic", "s", "p")
    assert out["cached"] and out["key"] == k and out["json"] == {"verdict": "reject"}
    assert out["cost"] == pytest.approx(cost_usd("deepseek-reasoner", 100, 50))
    # The same request with the new explicit names misses the legacy entry: a different (model, thinking) key.
    flash_on = CachedLLM(backend(no_network, model="deepseek-flash", thinking=True), cache_dir=tmp_path)
    flash_off = CachedLLM(backend(no_network, model="deepseek-flash"), cache_dir=tmp_path)
    keys = {flash_on.key("sceptic", "s", "p"), flash_off.key("sceptic", "s", "p"), k}
    assert len(keys) == 3 and flash_on.legacy_key("sceptic", "s", "p") is None
    assert reasoner.key("sceptic", "s", "p") == flash_on.key("sceptic", "s", "p")
    # A legacy name with the thinking flag overridden is a different request: no legacy lookup.
    override = CachedLLM(backend(no_network, model="deepseek-reasoner", thinking=False), cache_dir=tmp_path)
    assert override.legacy_key("sceptic", "s", "p") is None


def test_new_replies_are_stored_under_the_resolved_key_and_replayed(no_sleep, tmp_path):
    handler, bodies = recording([answer()])
    llm = CachedLLM(backend(handler, model="deepseek-chat"), cache_dir=tmp_path)
    first = llm.ask("analyst", "s", "p")
    again = CachedLLM(backend(handler, model="deepseek-flash", thinking=False), cache_dir=tmp_path).ask(
        "analyst", "s", "p")
    assert not first["cached"] and again["cached"] and first["key"] == again["key"] and len(bodies) == 1
    stored = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert stored["model"] == "deepseek-flash" and stored["thinking"] is False
    assert stored["requested_model"] == "deepseek-chat"
    assert first["key"] != old_key("deepseek-chat", "analyst", "s", "p")


# ---------------- prices ----------------

def test_peak_hours_and_peak_aware_cost():
    monday_2am = calendar.timegm((2026, 10, 12, 2, 30, 0))
    monday_5am = calendar.timegm((2026, 10, 12, 5, 0, 0))
    saturday_2am = calendar.timegm((2026, 10, 17, 2, 0, 0))
    assert is_peak(monday_2am) and not is_peak(monday_5am) and not is_peak(saturday_2am)
    off = cost_usd("deepseek-flash", 1_000_000, 1_000_000, monday_5am)
    assert off == pytest.approx(0.15 + 0.60)
    assert cost_usd("deepseek-flash", 1_000_000, 1_000_000, monday_2am) == pytest.approx(2 * off)
    assert cost_usd("deepseek-v4-pro", 1_000_000, 1_000_000, saturday_2am) == pytest.approx(0.66 + 1.98)
    # Legacy names keep the estimate used for the published runs, peak or not.
    assert cost_usd("deepseek-reasoner", 1_000_000, 0, monday_2am) == pytest.approx(0.27)
