"""Tool agent: as-of tools, cache, validation, sceptic and risk (no network)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from agents.llm_client import CachedLLM, StubBackend
from agents.tool_agent import PlainLLMAgent, ToolAgent, heuristic_reply, validate_proposal
from agents.tools import AsOfTools, numbers
from replay import Replay
from tests.test_agent_graph import HOME, NEWS_AT, TIP2, make_tables


def run(agent, tables=None):
    tables = tables or make_tables()
    decisions, fills = Replay(tables, agent.policy).run("2026-02-01", "2026-02-01")
    return decisions, fills, agent.traces


def tables_with_history():
    """make_tables plus a 24h-earlier quote so get_anchor has a true 24h mid."""
    t = make_tables()
    early = TIP2 - pd.Timedelta(hours=24)
    extra = []
    for ticker, bid, ask in ((HOME, 0.55, 0.57), ("SYN-g2-AWY", 0.43, 0.45)):
        for i, ts in enumerate(pd.date_range(early - pd.Timedelta(minutes=5), early + pd.Timedelta(minutes=5),
                                             freq="1min")):
            extra.append({"venue": "synthetic", "market_ticker": ticker, "ts": ts, "bid": bid, "ask": ask,
                          "volume": 100.0})
    t["prices"] = pd.concat([t["prices"], pd.DataFrame(extra)], ignore_index=True).sort_values("ts")
    return t


def view_at(tables, now=NEWS_AT):
    rp = Replay(tables, lambda *a: [])
    game = next(tables["games"][tables["games"].game_id == "g2"].itertuples(index=False))
    from replay import AsOf
    return AsOf(tables, now, rp.price_index), game, now


def agent(fn=heuristic_reply, sceptic=True, stake=20.0, cache_dir=None, anonymise=False):
    llm = CachedLLM(StubBackend(fn, model="test-stub"), cache_dir=cache_dir)
    return ToolAgent(llm, sceptic=sceptic, stake=stake, anonymise=anonymise,
                     forecaster=_forecaster(), players={7: "Test Player"})


def _forecaster():
    """Tiny stand-in for M4: home win falls by 0.10 when player 7 is out."""
    class F:
        def win(self, view, game, out=()):
            p = 0.55 - (0.10 if 7 in set(out) else 0.0)
            return {"p_home": p, "overrides": {"out": list(out)}}

        def __call__(self, view, game, markets, overrides):
            p = self.win(view, game, overrides.get("out", []))["p_home"]
            return {m.market_ticker: (p if m.team == game.home_team else 1 - p)
                    for m in markets.itertuples() if m.kind == "game"}
    return F()


# ---------------- tools ----------------

def test_tools_return_nothing_after_now():
    tables = tables_with_history()
    view, game, now = view_at(tables)
    tools = AsOfTools(view, game, now, forecaster=_forecaster(), players={7: "Test Player"})
    q = tools.get_quote("HOM")
    assert q["fresh"] and q["mid"] == pytest.approx(0.61)
    hist = tools.get_price_history("HOM", hours=3)
    assert hist["updates"] > 0 and hist["mid_now"] == q["mid"]
    a = tools.get_anchor("HOM")
    assert a["anchor_mid"] == pytest.approx(0.56) and a["move_since_anchor"] == pytest.approx(0.05)
    news = tools.get_news()
    assert news["count"] == 1 and news["items"][0]["player_id"] == 7
    assert news["items"][0]["minutes_ago"] >= 0
    m4 = tools.m4_win_prob([7])
    assert m4["before"]["HOM"] == pytest.approx(0.55) and m4["after"]["HOM"] == pytest.approx(0.45)
    assert tools.m4_win_prob([9999])["ignored"] == [9999]             # unknown player ignored
    fee = tools.fee(0.5)
    assert fee["fee_per_contract"] == pytest.approx(0.0175)
    e = tools.edge("AWY", 0.70)
    assert e["clears_min_edge"] and e["price"] == pytest.approx(0.40)  # yes on AWY at ask 0.40
    # A news item published after `now` is invisible.
    future = tables["news"].iloc[0].copy()
    future["news_id"], future["published_at"] = "n_future", now + pd.Timedelta(minutes=1)
    tables["news"] = pd.concat([tables["news"], pd.DataFrame([future])], ignore_index=True)
    view2, _, _ = view_at(tables)
    assert "n_future" not in {i["news_id"] for i in AsOfTools(view2, game, now).get_news()["items"]}
    assert set(AsOfTools(view2, game, now).view.news().news_id) == {"n_out"}


def test_tools_anonymise_labels():
    view, game, now = view_at(tables_with_history())
    tools = AsOfTools(view, game, now, anonymise=True, players={7: "LeBron"})
    assert tools.teams == ["TEAM_A", "TEAM_B"]
    news = tools.get_news()
    assert news["items"][0]["player"] == "Player 7"
    assert "LeBron" not in news["items"][0]["text"]


# ---------------- agent behaviour ----------------

def test_heuristic_agent_trades_when_edge_clears():
    """Stub plays anchor + M4 shift; with a 24h-early mid of 0.56 and a −0.10 home shift, AWY has edge."""
    tables = tables_with_history()
    # Home mid now 0.61, early 0.56; AWY ask 0.38. Anchor(AWY)≈0.44 + shift 0.10 = 0.54 → gap clears 4¢.
    a = agent(sceptic=False)
    decisions, fills, _ = run(a, tables)
    assert len(fills) == 1
    assert decisions.iloc[0].market_ticker == "SYN-g2-AWY" and decisions.iloc[0].side == "yes"
    assert a.traces[0]["status"] == "order"


def test_cache_hits_on_rerun(tmp_path):
    tables = tables_with_history()
    a1 = agent(sceptic=False, cache_dir=tmp_path)
    run(a1, tables)
    assert a1.llm.log and not any(c["cached"] for c in a1.llm.log)
    a2 = agent(sceptic=False, cache_dir=tmp_path)
    run(a2, tables)
    assert a2.llm.log and all(c["cached"] for c in a2.llm.log)
    assert a2.llm.backend.calls == 0                                  # stub never contacted


class _QuotaError(Exception):
    code = 429


def _gemini(errors, monkeypatch):
    """GeminiBackend without a network client: the first len(errors) calls raise, then it answers."""
    from types import SimpleNamespace

    from agents import llm_client
    sleeps = []
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    b = llm_client.GeminiBackend.__new__(llm_client.GeminiBackend)
    b.model, b.max_retries, b.min_interval, b._last = "gemini-test", 3, 0.0, 0.0
    b._lock = __import__("threading").Lock()
    b._config = lambda system: None
    queue = list(errors)

    def generate_content(**kw):
        if queue:
            raise _QuotaError(queue.pop(0))
        return SimpleNamespace(text='{"ok": true}', usage_metadata=SimpleNamespace(prompt_token_count=3,
                                                                                   candidates_token_count=2))
    b.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return b, sleeps


def test_rate_limit_is_retried_after_the_hinted_wait(monkeypatch):
    b, sleeps = _gemini(["429 RESOURCE_EXHAUSTED. Please retry in 7.5s."], monkeypatch)
    text, usage = b.generate("system", "prompt")
    assert text == '{"ok": true}' and usage["tokens_in"] == 3
    assert sleeps == [8.5]


def test_spent_daily_quota_fails_fast_and_is_not_cached(monkeypatch, tmp_path):
    b, sleeps = _gemini(["429 RESOURCE_EXHAUSTED. Please retry in 19h21m10.7s."], monkeypatch)
    llm = CachedLLM(b, cache_dir=tmp_path)
    out = llm.ask("analyst", "system", "prompt")
    assert out["error"] and out["json"] is None and not sleeps
    assert not list(tmp_path.rglob("*.json"))


def test_invalid_llm_json_is_a_pass():
    a = agent(fn=lambda *x: "not json at all", sceptic=False)
    decisions, fills, _ = run(a, tables_with_history())
    assert decisions.empty and fills.empty
    assert a.traces[0]["status"] == "invalid" and a.traces[0]["reason"] == "invalid_json"


def test_cited_number_mismatch_is_caught():
    def liar(role, system, prompt, meta):
        if role != "analyst":
            return heuristic_reply(role, system, prompt, meta)
        out = heuristic_reply(role, system, prompt, meta)
        if "final" in out:
            out["final"]["citations"][0]["value"] = 0.99              # not in any tool output
            out["final"]["estimate_p"] = 0.99
        return out
    a = agent(fn=liar, sceptic=False)
    decisions, fills, _ = run(a, tables_with_history())
    assert fills.empty
    assert a.traces[0]["status"] == "invalid" and a.traces[0]["reason"] in ("number_mismatch", "ungrounded_estimate")


def test_validate_proposal_catches_mismatched_citation():
    view, game, now = view_at(tables_with_history())
    tools = AsOfTools(view, game, now, forecaster=_forecaster())
    results = {"c1": {"tool": "get_anchor", "args": {"team": "HOM"}, "output": tools.get_anchor("HOM")}}
    ok, why, _ = validate_proposal({"decision": "buy", "team": "HOM", "estimate_p": 0.56,
                                    "rationale": "Anchor 0.56.",
                                    "citations": [{"call_id": "c1", "field": "anchor_mid", "value": 0.99}]},
                                   tools, results, view)
    assert not ok and why == "number_mismatch"


def test_sceptic_reject_blocks_order():
    def reject(role, system, prompt, meta):
        if role == "sceptic":
            return {"verdict": "reject", "reason": "Already in the price."}
        return heuristic_reply(role, system, prompt, meta)
    a = agent(fn=reject, sceptic=True)
    decisions, fills, _ = run(a, tables_with_history())
    assert fills.empty and decisions.empty
    t = a.traces[0]
    assert t["status"] == "sceptic_reject" and "candidate" in t
    assert t["sceptic"]["verdict"] == "reject"


def test_sceptic_approve_allows_order():
    # Heuristic sceptic rejects only when the mid has moved ≥ 3¢ toward the trade since the anchor.
    # On this slate AWY's mid fell (0.44 → 0.39), so the sceptic approves.
    a = agent(sceptic=True)
    decisions, fills, _ = run(a, tables_with_history())
    assert len(fills) == 1 and a.traces[0]["status"] == "order"
    assert a.traces[0]["sceptic"]["verdict"] == "approve"


def test_risk_caps_still_apply():
    a = agent(sceptic=False, stake=500.0)                              # over the $50 order cap
    decisions, fills, _ = run(a, tables_with_history())
    assert fills.empty
    assert a.traces[0]["status"] == "risk_block" and a.traces[0]["reason"] == "over_order_cap"


def test_plain_llm_uses_same_trading_rule():
    def plain(role, system, prompt, meta):
        assert role == "plain"
        return {"p_home": 0.30, "rationale": "Home looks weak after the news."}   # AWY at 0.70 clears
    llm = CachedLLM(StubBackend(plain), cache_dir=None)
    a = PlainLLMAgent(llm, forecaster=_forecaster())
    decisions, fills, _ = run(a, tables_with_history())
    assert len(fills) == 1 and decisions.iloc[0].market_ticker == "SYN-g2-AWY"


def test_one_position_per_game():
    a = agent(sceptic=False)
    decisions, fills, _ = run(a, tables_with_history())
    assert len(fills) == 1
    assert a.traces[1]["status"] == "held"                            # LEAD decision: already held
