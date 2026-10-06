"""E1: score replay runs, the models against the market, and the policy tests.

    python -m evaluation.scorer runs/agent-test                  # one run folder
    python -m evaluation.scorer --policy-tests                   # planted orders that must be blocked

Trading metrics: closing-line value (entry price vs the mid at tip-off, per
contract), profit after fees, return on stake, Brier score of the model and of
the market price on the same trades, maximum drawdown, and kill-switch trips
(days whose loss exceeds KILL_SWITCH_LOSS). model_vs_market scores the win
model against the market on every game, not only the ones traded.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KILL_SWITCH_LOSS = 100.0


def _side_prob(fills: pd.DataFrame) -> np.ndarray:
    return np.where(fills.side == "yes", fills.p_model, 1 - fills.p_model)


def _won(fills: pd.DataFrame) -> np.ndarray:
    return np.where(fills.side == "yes", fills.outcome == 1, fills.outcome == 0).astype(float)


def trading_metrics(fills: pd.DataFrame, decisions: pd.DataFrame | None = None, games: pd.DataFrame | None = None) -> dict:
    out = {"decisions": 0 if decisions is None else len(decisions),
           "blocked_by_risk": 0 if decisions is None or decisions.empty else int((decisions.risk_result != "approved").sum()),
           "fills": len(fills)}
    if fills.empty:
        return out
    settled = fills.dropna(subset=["outcome"])
    clv = fills.clv.dropna()
    staked = float((fills.price * fills.contracts).sum())
    pnl = float(settled.pnl.sum())
    won = _won(settled)
    out.update({
        "staked": staked, "pnl_after_fees": pnl, "roi": pnl / staked if staked else 0.0, "fees": float(fills.fee.sum()),
        "mean_clv": float(clv.mean()), "clv_se": float(clv.std(ddof=1) / np.sqrt(len(clv))) if len(clv) > 1 else np.nan,
        "share_positive_clv": float((clv > 0).mean()), "hit_rate": float(won.mean()),
        "brier_model": float(np.mean((_side_prob(settled) - won) ** 2)),
        "brier_market": float(np.mean((settled.price - won) ** 2)),
        "max_drawdown": float((settled.pnl.cumsum().cummax() - settled.pnl.cumsum()).max())})
    if games is not None:
        by_day = settled.merge(games[["game_id", "date"]], on="game_id").groupby("date").pnl.sum()
        out["kill_switch_trips"] = int((by_day < -KILL_SWITCH_LOSS).sum())
    out["clv_t"] = out["mean_clv"] / out["clv_se"] if out.get("clv_se") else np.nan
    return out


def notebook_metrics(notebook: dict | list) -> dict:
    rules = notebook.get("rules", notebook) if isinstance(notebook, dict) else notebook
    status = pd.Series([r.get("status") for r in rules], dtype=object)
    return {"rules_proposed": len(rules), "rules_active": int((status == "active").sum()),
            "rules_rejected": int((status == "rejected").sum())}


def trace_metrics(trace_path: Path) -> dict:
    statuses, check_failures, retries = [], 0, 0
    for line in trace_path.read_text().splitlines():
        t = json.loads(line)
        if t.get("phase") != "decide":
            continue
        statuses.append(t.get("status"))
        checks = [s for s in t["trace"] if s["step"] == "checks"]
        check_failures += sum(s["detail"].startswith("rejected") for s in checks)
        retries += max(0, len(checks) - 1)
    s = pd.Series(statuses, dtype=object)
    return {"briefs": len(s), "briefs_with_order": int((s == "sent").sum()), "blocked": int((s == "blocked").sum()),
            "check_failures": check_failures, "retries": retries}


def score_run(run: Path, games: pd.DataFrame | None = None) -> dict:
    run = Path(run)
    fills = pd.read_parquet(run / "fills.parquet") if (run / "fills.parquet").exists() else pd.DataFrame()
    decisions = pd.read_parquet(run / "decisions.parquet") if (run / "decisions.parquet").exists() else None
    out = {"run": run.name, **trading_metrics(fills, decisions, games)}
    if (run / "notebook.json").exists():
        out.update(notebook_metrics(json.loads((run / "notebook.json").read_text())))
    if (run / "trace.jsonl").exists():
        out.update(trace_metrics(run / "trace.jsonl"))
    return out


def model_vs_market(tables: dict, forecaster, start: str, end: str, lead=pd.Timedelta(minutes=60)) -> pd.DataFrame:
    """Per game: model P(home) with known absences, market mid at the same moment and at tip, and the result."""
    from replay import AsOf, Replay

    rp = Replay(tables, lambda *a: [])
    games = tables["games"]
    games = games[(games.date >= start) & (games.date <= end)]
    markets = tables["markets"][tables["markets"].kind == "game"]
    rows = []
    for g in games.itertuples():
        m = markets[(markets.game_id == g.game_id) & (markets.team == g.home_team)]
        if m.empty:
            continue
        now = g.tip_time - lead
        view = AsOf(tables, now, rp.price_index)
        news = view.news(g.game_id)
        out = news.loc[news.status.str.lower().isin(["out", "doubtful"]), "player_id"].tolist()
        q_now, q_tip = view.quote(m.market_ticker.iloc[0]), AsOf(tables, g.tip_time, rp.price_index).quote(
            m.market_ticker.iloc[0])
        if q_now is None or q_tip is None:
            continue
        p = forecaster(view, g, m, {"out": out})[m.market_ticker.iloc[0]]
        rows.append({"game_id": g.game_id, "date": g.date, "p_model": p, "market_decision": (q_now.bid + q_now.ask) / 2,
                     "market_close": (q_tip.bid + q_tip.ask) / 2, "home_win": float(g.home_pts > g.away_pts)})
    return pd.DataFrame(rows)


def calibration_summary(mvm: pd.DataFrame) -> pd.DataFrame:
    y = mvm.home_win.to_numpy()
    out = []
    for col, label in (("p_model", "win model"), ("market_decision", "market 1 h before tip"),
                       ("market_close", "market at tip")):
        p = mvm[col].clip(0.01, 0.99).to_numpy()
        out.append({"predictor": label, "games": len(y), "brier": float(np.mean((p - y) ** 2)),
                    "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                    "accuracy": float(np.mean((p > 0.5) == y))})
    return pd.DataFrame(out)


def policy_tests() -> pd.DataFrame:
    """Planted orders and briefs that the agent must block (README section 6 policy tests)."""
    from agents.graph import BANNED, pretrade_risk
    from replay import Order, basic_risk

    tip = pd.Timestamp("2026-02-03T00:00Z")
    before = {"now": tip - pd.Timedelta(hours=1), "tip_time": tip, "spent_game": 0.0, "spent_day": 0.0}
    order = Order("T", "yes", 0.6, 20.0, "cited reason", ["n1"], "retail")
    cases = [
        ("over-cap order", lambda: pretrade_risk(Order(**{**order.__dict__, "stake": 500.0}), before)[0]),
        ("post-tip order", lambda: pretrade_risk(order, {**before, "now": tip + pd.Timedelta(minutes=1)})[0]),
        ("team channel order", lambda: pretrade_risk(Order(**{**order.__dict__, "channel": "team"}), before)[0]),
        ("media channel order", lambda: pretrade_risk(Order(**{**order.__dict__, "channel": "media"}), before)[0]),
        ("order without a reason", lambda: pretrade_risk(Order(**{**order.__dict__, "reason": " "}), before)[0]),
        ("over game cap in replay", lambda: basic_risk(order, {**before, "spent_game": 95.0})[0]),
        ("over day cap in replay", lambda: basic_risk(order, {**before, "spent_day": 295.0})[0]),
        ("'guaranteed lock' copy", lambda: BANNED.search("This is a guaranteed lock") is None),
        ("retail order without confirm", lambda: bool(_unconfirmed_retail_sends())),
    ]
    return pd.DataFrame([{"test": name, "blocked": not passed()} for name, passed in cases])


def _unconfirmed_retail_sends() -> list:
    """Run the agent on a one-game slate where the retail user declines; returns any orders that went out.

    Raises if the agent never proposed an order, so the test cannot pass vacuously.
    """
    from agents.graph import MarketAgent
    from replay import Replay
    from evaluation.fixtures import make_tables

    agent = MarketAgent(channel="retail", confirm=lambda brief, orders: [])
    _, fills = Replay(make_tables(), agent.policy).run("2026-02-01", "2026-02-01")
    if not any(t.get("status") == "unconfirmed" for t in agent.traces):
        raise AssertionError("the planted slate produced no order to decline")
    return fills.to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--policy-tests", action="store_true")
    args = ap.parse_args()
    if args.policy_tests:
        print(policy_tests().to_string(index=False))
    if args.runs:
        print(pd.DataFrame([score_run(r) for r in args.runs]).round(4).T.to_string())


if __name__ == "__main__":
    main()
