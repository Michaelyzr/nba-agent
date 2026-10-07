"""Placebo gate audit: reviewer rules vs random templates and random slices.

    python -m evaluation.gate_placebo                 # uses runs/walkforward/no_learning if present
    python -m evaluation.gate_placebo --replay        # rebuild the no-learning base run first

Pre-registered in docs/preregistration_gate.md. At each review point the script
gates (a) the reviewer's pick, (b) every template and (c) K random-slice rules
under legacy, split and split-edge, without replaying: skip_market and min_edge
only remove trades, so applying a rule to the base fills is exact. Writes
evaluation/results/gate_placebo.{csv,md}.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from agents.graph import (GATE_DAYS, GATE_METRICS, GATE_MODES, REVIEW_TEMPLATES, SELECT_DAYS, MarketAgent,
                          gate_totals, gate_windows, judge, market_days)
from agents.notebook import LIMITS, Notebook
from evaluation.stats import REPS, SEED
from replay import RUNS

RESULTS = Path(__file__).resolve().parent / "results"
START, END = "2025-11-01", "2026-04-12"
BASE_RUN = RUNS / "walkforward" / "no_learning"
PLACEBO_K = 5
REVIEW_EVERY = 7


def wilson(successes: int, n: int, z=1.96) -> tuple:
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def load_base(path: Path, agent: MarketAgent | None = None) -> tuple:
    fills = pd.read_parquet(path / "fills.parquet")
    situations = {}
    if agent is not None:
        situations = dict(agent.situations)
    elif (path / "situations.json").exists():
        raw = json.loads((path / "situations.json").read_text())
        situations = {(t, pd.Timestamp(a)): s for (t, a), s in (((k.split("|", 1)[0], k.split("|", 1)[1]), v)
                                                                for k, v in raw.items())}
    return fills, situations


def save_situations(path: Path, situations: dict):
    body = {f"{t}|{pd.Timestamp(a).isoformat()}": s for (t, a), s in situations.items()}
    (path / "situations.json").write_text(json.dumps(body, default=float))


def ensure_base(replay: bool, workers: int = 1):
    """No-learning walk-forward fills with decision-time situations for the placebo audit."""
    from agents.graph import save_run
    from evaluation.walkforward import WalkForwardForecaster, train_win_models, WINDOWS, SEASON_START
    from forecast.api import Forecaster
    from replay import Replay, load_tables

    if not replay and (BASE_RUN / "fills.parquet").exists() and (BASE_RUN / "situations.json").exists():
        return load_base(BASE_RUN)
    tables = load_tables()
    base = Forecaster.load()
    starts = [SEASON_START] + [w[1] for w in WINDOWS]
    models, _ = train_win_models(starts)
    forecaster = WalkForwardForecaster(models, base.play_model, base.points_model)
    agent = MarketAgent(forecaster=forecaster, learn=False, gate="legacy")
    rp = Replay(tables, agent.policy)
    decisions, fills = rp.run(START, END)
    save_run("walkforward/no_learning_placebo", decisions, fills, agent)
    out = RUNS / "walkforward" / "no_learning_placebo"
    save_situations(out, agent.situations)
    # Prefer the official no_learning folder when it already matches; keep a parallel copy for situations.
    if not (BASE_RUN / "fills.parquet").exists():
        save_run("walkforward/no_learning", decisions, fills, agent)
        save_situations(BASE_RUN, agent.situations)
        return fills, agent.situations
    return fills, agent.situations


def apply_rule(fills: pd.DataFrame, situations: dict, rule: dict) -> pd.DataFrame:
    """Fills that survive the rule (skip_market / min_edge only remove trades)."""
    if fills.empty:
        return fills
    sit = pd.DataFrame([situations.get((t, a), {}) for t, a in zip(fills.market_ticker, fills.as_of)],
                       index=fills.index)
    from agents.notebook import matches
    keep = []
    for i, row in fills.iterrows():
        s = sit.loc[i].to_dict() if len(sit) else {}
        s = {k: (None if pd.isna(v) else v) for k, v in s.items()}
        if rule["do"]["action"] == "skip_market" and matches(rule, s):
            keep.append(False)
        elif rule["do"]["action"] == "min_edge" and s.get("gap", 0) < rule["do"]["params"]["edge"]:
            keep.append(False)
        else:
            keep.append(True)
    return fills.loc[keep]


def day_fills(fills: pd.DataFrame, games: pd.DataFrame, days) -> pd.DataFrame:
    return fills[fills.game_id.isin(games.loc[games.date.isin(days), "game_id"])]


def make_rp(tables, fills):
    from types import SimpleNamespace
    return SimpleNamespace(t=tables, fills=fills.to_dict("records") if len(fills) else [])


def reviewer_pick(agent: MarketAgent, fills: pd.DataFrame, situations: dict, day: str, mode: str, rp) -> dict | None:
    agent.situations = situations
    agent.gate_mode = mode
    recent = agent.selection(rp, fills, day, mode)
    if recent.empty:
        return None
    return agent._review_offline(recent)


def random_slice_rule(seed: int, share: float, idx: int) -> dict:
    rng = np.random.default_rng(seed + 17 * idx)
    width = float(np.clip(share, 0.05, 0.5))
    lo = float(rng.uniform(0, 1 - width))
    return {"rule_id": f"placebo-{idx}", "version": 1, "kind": "trader", "status": "proposed",
            "when": {"market_kind": "game", "placebo_bucket_min": lo, "placebo_bucket_max": lo + width},
            "do": {"action": "skip_market", "params": {}}, "blame_category": "placebo",
            "rationale": f"random slice [{lo:.3f}, {lo + width:.3f})", "priority": 1,
            "expires_after_days": LIMITS["default_expiry_days"]}


def evaluate_rule(fills, situations, games, days, rule, mode) -> dict:
    without = day_fills(fills, games, days)
    with_rule = apply_rule(without, situations, rule)
    return judge(without, with_rule, mode)


def forward_clv(fills, situations, games, days, rule) -> float:
    window = day_fills(fills, games, days)
    before = gate_totals(window)["clv_dollars"]
    after = gate_totals(apply_rule(window, situations, rule))["clv_dollars"]
    return after - before


def run_audit(fills: pd.DataFrame, situations: dict, tables: dict) -> pd.DataFrame:
    games = tables["games"]
    rp = make_rp(tables, fills)
    agent = MarketAgent(learn=False)
    agent.situations = situations
    all_days = market_days(rp, END)
    # Need SELECT_DAYS + GATE_DAYS history before a review point.
    start_i = SELECT_DAYS + GATE_DAYS
    review_days = all_days[start_i::REVIEW_EVERY]
    rows = []
    for day in review_days:
        for mode in GATE_MODES:
            pick = reviewer_pick(agent, fills, situations, day, mode, rp)
            _, test_days = gate_windows(rp, day, mode)
            if len(test_days) < 2:
                continue
            fwd_days = [d for d in all_days if d > day][:SELECT_DAYS]
            # Reviewer's pick
            if pick is not None:
                g = evaluate_rule(fills, situations, games, test_days, pick, mode)
                rows.append({"review_day": day, "mode": mode, "kind": "reviewer", "rule": json.dumps(pick["when"]),
                             "accepted": int(g["result"] == "accepted"), **{k: g[k] for k in
                             ("cases", "clv_dollars_before", "clv_dollars_after", "edge_dollars_before",
                              "edge_dollars_after", "before", "after")},
                             "forward_clv_dollars": forward_clv(fills, situations, games, fwd_days, pick)})
            # Every template
            for when, do, blame, says in REVIEW_TEMPLATES:
                rule = {"when": {"market_kind": "game", **when}, "do": do, "blame_category": blame}
                g = evaluate_rule(fills, situations, games, test_days, rule, mode)
                rows.append({"review_day": day, "mode": mode, "kind": "template", "rule": json.dumps(rule["when"]),
                             "accepted": int(g["result"] == "accepted"), **{k: g[k] for k in
                             ("cases", "clv_dollars_before", "clv_dollars_after", "edge_dollars_before",
                              "edge_dollars_after", "before", "after")},
                             "forward_clv_dollars": forward_clv(fills, situations, games, fwd_days, rule)})
            # Random slices: width = reviewer hit rate on selection, else 0.25
            select_days, _ = gate_windows(rp, day, mode)
            sel = day_fills(fills, games, select_days or []) if select_days else fills
            share = 0.25
            if pick is not None and len(sel):
                kept = apply_rule(sel, situations, pick)
                share = 1 - len(kept) / len(sel) if len(sel) else 0.25
            for i in range(PLACEBO_K):
                rule = random_slice_rule(SEED, share, hash((day, mode, i)) % 10_000)
                g = evaluate_rule(fills, situations, games, test_days, rule, mode)
                rows.append({"review_day": day, "mode": mode, "kind": "random_slice",
                             "rule": json.dumps(rule["when"]), "accepted": int(g["result"] == "accepted"),
                             **{k: g[k] for k in ("cases", "clv_dollars_before", "clv_dollars_after",
                                                  "edge_dollars_before", "edge_dollars_after", "before", "after")},
                             "forward_clv_dollars": forward_clv(fills, situations, games, fwd_days, rule)})
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> str:
    lines = ["# Placebo gate audit", "",
             f"Base: no-learning anchor agent, {START}–{END}. Review every {REVIEW_EVERY}th market day "
             f"after {SELECT_DAYS + GATE_DAYS} days of history. At each point: the reviewer's pick, all "
             f"{len(REVIEW_TEMPLATES)} templates, and {PLACEBO_K} random-slice rules, gated under "
             f"{', '.join(GATE_MODES)}. Pass-rate CIs are Wilson 95%. Seed {SEED}.", "",
             "## Pass rates", "",
             "| Mode | Kind | N | Passes | Rate [95% CI] |", "| --- | --- | --- | --- | --- |"]
    for mode in GATE_MODES:
        for kind in ("reviewer", "template", "random_slice"):
            part = frame[(frame.mode == mode) & (frame.kind == kind)]
            n, k = len(part), int(part.accepted.sum()) if len(part) else 0
            lo, hi = wilson(k, n)
            rate = "–" if n == 0 else f"{k / n:.1%} [{lo:.1%}, {hi:.1%}]"
            lines.append(f"| {mode} | {kind} | {n} | {k} | {rate} |")
    lines += ["", "## H1 / H2 checks", "",
              "H1: fewer reviewer passes under split than under legacy.",
              "H2: under legacy, reviewer pass rate ≈ template / random_slice pass rate "
              "(learning mainly reduces exposure).", ""]
    for mode in GATE_MODES:
        rev = frame[(frame.mode == mode) & (frame.kind == "reviewer")]
        rnd = frame[(frame.mode == mode) & (frame.kind == "random_slice")]
        tmpl = frame[(frame.mode == mode) & (frame.kind == "template")]
        def rate(p):
            return np.nan if p.empty else p.accepted.mean()
        lines.append(f"- **{mode}**: reviewer {rate(rev):.1%} vs templates {rate(tmpl):.1%} vs "
                     f"random slices {rate(rnd):.1%}.")
        if len(rev):
            acc = rev[rev.accepted == 1].forward_clv_dollars
            rej = rev[rev.accepted == 0].forward_clv_dollars
            lines.append(f"  Forward CLV $ (next {SELECT_DAYS} days): accepted mean "
                         f"{acc.mean():+.2f} (n={len(acc)}), rejected mean {rej.mean():+.2f} (n={len(rej)}).")
    # Bootstrap CI on reviewer − random_slice pass-rate difference under legacy
    lines += ["", "## Bootstrap: reviewer − random_slice pass rate (legacy)", ""]
    legacy_days = sorted(frame[frame.mode == "legacy"].review_day.unique())
    if legacy_days:
        rng = np.random.default_rng(SEED)
        diffs = []
        for _ in range(REPS):
            days = rng.choice(legacy_days, size=len(legacy_days), replace=True)
            r = frame[(frame.mode == "legacy") & (frame.kind == "reviewer") & (frame.review_day.isin(days))]
            s = frame[(frame.mode == "legacy") & (frame.kind == "random_slice") & (frame.review_day.isin(days))]
            if r.empty or s.empty:
                continue
            diffs.append(r.accepted.mean() - s.accepted.mean())
        if diffs:
            lo, hi = np.quantile(diffs, [0.025, 0.975])
            point = (frame[(frame.mode == "legacy") & (frame.kind == "reviewer")].accepted.mean()
                     - frame[(frame.mode == "legacy") & (frame.kind == "random_slice")].accepted.mean())
            lines.append(f"Difference {point:+.1%} [{lo:+.1%}, {hi:+.1%}]. "
                         "CI includes 0 ⇒ cannot reject H2 (gate does not prefer the reviewer over noise).")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="rebuild the no-learning base run with situations")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    from replay import load_tables
    tables = load_tables()
    # Prefer situations sibling; fall back to replaying
    fills, situations = None, {}
    for path in (BASE_RUN, RUNS / "walkforward" / "no_learning_placebo"):
        if (path / "fills.parquet").exists() and (path / "situations.json").exists() and not args.replay:
            fills, situations = load_base(path)
            print(f"loaded base from {path}: {len(fills)} fills, {len(situations)} situations")
            break
    if fills is None:
        print("building no-learning base with situations (one walk-forward pass)")
        fills, situations = ensure_base(replay=True)
    # Situations keys may be string timestamps; normalise to match fills.as_of
    fills = fills.copy()
    fills["as_of"] = pd.to_datetime(fills.as_of)
    situations = {(t, pd.Timestamp(a)): s for (t, a), s in situations.items()}
    # Rebuild situations from a short agent pass if empty
    if not situations:
        print("situations missing; replaying no-learning to capture them")
        fills, situations = ensure_base(replay=True)
        fills["as_of"] = pd.to_datetime(fills.as_of)
    frame = run_audit(fills, situations, tables)
    frame.to_csv(RESULTS / "gate_placebo.csv", index=False)
    md = summarise(frame)
    (RESULTS / "gate_placebo.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
