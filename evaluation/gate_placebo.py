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
# Deadline default: test window. Pass --start/--end for the primary Nov–Apr window.
START, END = "2026-02-01", "2026-04-12"
BASE_RUN = RUNS / "gate_audit" / "test-split_nolearn"
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


KEYS = ("cases", "clv_dollars_before", "clv_dollars_after", "edge_dollars_before", "edge_dollars_after",
        "before", "after")


def legacy_selection(fills: pd.DataFrame, games: pd.DataFrame, day: str) -> pd.DataFrame:
    """Legacy reviewer input: settled trades in the REVIEW_WINDOW ending on `day` (never later fills)."""
    from agents.graph import REVIEW_WINDOW
    known = day_fills(fills, games, games.loc[games.date <= day, "date"].unique())
    if known.empty:
        return known
    as_of = pd.to_datetime(known.as_of)
    return known[as_of >= as_of.max() - REVIEW_WINDOW]


def run_audit(fills: pd.DataFrame, situations: dict, tables: dict) -> pd.DataFrame:
    games = tables["games"]
    rp = make_rp(tables, fills)
    agent = MarketAgent(learn=False)
    agent.situations = situations
    all_days = [d for d in market_days(rp, END) if d >= START]
    # Review points need SELECT_DAYS + GATE_DAYS market days of history inside the window.
    review_days = all_days[SELECT_DAYS + GATE_DAYS::REVIEW_EVERY]
    rows = []
    for r_i, day in enumerate(review_days):
        fwd_days = [d for d in all_days if d > day][:SELECT_DAYS]
        for m_i, mode in enumerate(GATE_MODES):
            select_days, test_days = gate_windows(rp, day, mode)
            if len(test_days) < 2:
                continue
            sel = legacy_selection(fills, games, day) if mode == "legacy" else day_fills(fills, games, select_days)
            agent.gate_mode = mode
            pick = agent._review_offline(sel) if len(sel) else None

            def add(kind, rule):
                g = evaluate_rule(fills, situations, games, test_days, rule, mode)
                rows.append({"review_day": day, "mode": mode, "kind": kind, "rule": json.dumps(rule["when"]),
                             "action": rule["do"]["action"], "accepted": int(g["result"] == "accepted"),
                             **{k: g[k] for k in KEYS},
                             "delta_clv_dollars": g["clv_dollars_after"] - g["clv_dollars_before"],
                             "forward_clv_dollars": forward_clv(fills, situations, games, fwd_days, rule)})

            if pick is not None:
                add("reviewer", pick)
            for when, do, blame, _ in REVIEW_TEMPLATES:
                add("template", {"when": {"market_kind": "game", **when}, "do": do, "blame_category": blame})
            share = 0.25
            if pick is not None and len(sel):
                share = 1 - len(apply_rule(sel, situations, pick)) / len(sel)
            for i in range(PLACEBO_K):
                add("random_slice", random_slice_rule(SEED, share, 1000 * r_i + 100 * m_i + i))
    return pd.DataFrame(rows)


def _boot_points(frame: pd.DataFrame, stat, reps=REPS, seed=SEED) -> tuple:
    """Point estimate and 95% percentile CI of stat(frame), resampling review points with replacement."""
    days = sorted(frame.review_day.unique())
    groups = {d: g for d, g in frame.groupby("review_day")}
    point = stat(frame)
    if not days:
        return point, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        pick = rng.choice(len(days), size=len(days), replace=True)
        v = stat(pd.concat([groups[days[j]] for j in pick], ignore_index=True))
        if not pd.isna(v):
            draws.append(v)
    if not draws:
        return point, np.nan, np.nan
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return point, float(lo), float(hi)


def _rate(frame, mode, kind):
    part = frame[(frame["mode"] == mode) & (frame.kind == kind)]
    return np.nan if part.empty else float(part.accepted.mean())


def summarise(frame: pd.DataFrame) -> tuple:
    """Markdown report and a tidy table of the bootstrap comparisons."""
    n_points = frame.review_day.nunique() if len(frame) else 0
    lines = ["# Placebo gate audit", "",
             f"Pre-registered in `docs/preregistration_gate.md` (H1, H2). Base: no-learning market-anchor agent, "
             f"{START}–{END} (**deadline deviation:** test window, not the walk-forward season). Review every "
             f"{REVIEW_EVERY}th market day after {SELECT_DAYS + GATE_DAYS} market days of history "
             f"({n_points} review points). At each point: the reviewer's own pick, all {len(REVIEW_TEMPLATES)} "
             f"templates (= exact pass rate of a uniformly random template) and {PLACEBO_K} random-slice rules "
             f"(seeded hash buckets, width = share of selection trades the reviewer's rule hit, else 0.25), each "
             f"gated under {', '.join(GATE_MODES)}. Rules only remove trades, so they are applied to the base fills "
             f"exactly, without replaying. Pass-rate CIs: Wilson 95%. Differences and CLV $ effects: bootstrap over "
             f"review points, {REPS} replicates, seed {SEED}.", "",
             "## Pass rates", "",
             "| Gate | Rule kind | Rules gated | Accepted | Pass rate [95% CI] | Mean Δ CLV $ on test days | "
             "Mean forward Δ CLV $ (next 7 days) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for mode in GATE_MODES:
        for kind in ("reviewer", "template", "random_slice"):
            part = frame[(frame["mode"] == mode) & (frame.kind == kind)]
            n, k = len(part), int(part.accepted.sum()) if len(part) else 0
            lo, hi = wilson(k, n)
            rate = "–" if n == 0 else f"{k / n:.1%} [{lo:.1%}, {hi:.1%}]"
            dlt = "–" if n == 0 else f"{part.delta_clv_dollars.mean():+.2f}"
            fwd = "–" if n == 0 else f"{part.forward_clv_dollars.mean():+.2f}"
            lines.append(f"| {mode} | {kind} | {n} | {k} | {rate} | {dlt} | {fwd} |")

    comps = []
    def comp(label, stat, kind="rate"):
        est, lo, hi = _boot_points(frame, stat)
        comps.append({"comparison": label, "kind": kind, "estimate": est, "ci_low": lo, "ci_high": hi})

    comp("H1: reviewer pass rate, split − legacy", lambda f: _rate(f, "split", "reviewer") - _rate(f, "legacy", "reviewer"))
    comp("H1: reviewer pass rate, split-edge − legacy",
         lambda f: _rate(f, "split-edge", "reviewer") - _rate(f, "legacy", "reviewer"))
    for mode in GATE_MODES:
        comp(f"H2 ({mode}): reviewer − random template pass rate",
             lambda f, m=mode: _rate(f, m, "reviewer") - _rate(f, m, "template"))
        comp(f"H2 ({mode}): reviewer − random slice pass rate",
             lambda f, m=mode: _rate(f, m, "reviewer") - _rate(f, m, "random_slice"))
    comp("H2: random-slice pass rate, split − legacy",
         lambda f: _rate(f, "split", "random_slice") - _rate(f, "legacy", "random_slice"))
    comp("H2: random-slice pass rate, split-edge − legacy",
         lambda f: _rate(f, "split-edge", "random_slice") - _rate(f, "legacy", "random_slice"))

    def fwd_gap(f, m):
        part = f[f["mode"] == m]
        a, r = part[part.accepted == 1].forward_clv_dollars, part[part.accepted == 0].forward_clv_dollars
        return np.nan if a.empty or r.empty else float(a.mean() - r.mean())

    for mode in GATE_MODES:
        comp(f"Forward CLV $, accepted − rejected rules ({mode}, all kinds)", lambda f, m=mode: fwd_gap(f, m), "$")
    for mode in GATE_MODES:
        comp(f"Δ CLV $ on test days of accepted reviewer rules ({mode})",
             lambda f, m=mode: (lambda p: np.nan if p.empty else float(p.delta_clv_dollars.mean()))(
                 f[(f["mode"] == m) & (f.kind == "reviewer") & (f.accepted == 1)]), "$")
    comps = pd.DataFrame(comps)

    lines += ["", "## Bootstrap comparisons", "", "| Comparison | Estimate [95% CI] |", "| --- | --- |"]
    for c in comps.itertuples():
        if pd.isna(c.estimate):
            val = "– (no cases)"
        elif c.kind == "rate":
            val = f"{c.estimate:+.1%} [{c.ci_low:+.1%}, {c.ci_high:+.1%}]"
        else:
            val = f"{c.estimate:+.2f} [{c.ci_low:+.2f}, {c.ci_high:+.2f}]"
        lines.append(f"| {c.comparison} | {val} |")
    lines += ["", "Reading the table: H1 is supported if the split − legacy reviewer CI lies below 0. H2 is "
              "supported if the reviewer − random CIs include 0 under legacy. If the accepted − rejected forward "
              "CLV $ CI includes 0, the gate cannot tell lessons from noise.", ""]
    return "\n".join(lines) + "\n", comps


def capture_situations(start: str, end: str) -> tuple:
    """No-learning replay over [start, end] that records decision-time situations."""
    from agents.graph import MarketAgent, save_run
    from forecast.api import Forecaster
    from replay import Replay, load_tables

    tables = load_tables()
    agent = MarketAgent(forecaster=Forecaster.load(), learn=False, gate="legacy")
    rp = Replay(tables, agent.policy)
    decisions, fills = rp.run(start, end)
    out = RUNS / "gate_placebo" / "base"
    save_run("gate_placebo/base", decisions, fills, agent)
    save_situations(out, agent.situations)
    return fills, agent.situations


def main():
    global START, END, BASE_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="rebuild the no-learning base run with situations")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()
    START, END = args.start, args.end
    RESULTS.mkdir(parents=True, exist_ok=True)
    from replay import load_tables
    tables = load_tables()
    # Prefer situations sibling; fall back to replaying the requested window
    fills, situations = None, {}
    for path in (BASE_RUN, RUNS / "gate_placebo" / "base",
                 RUNS / "walkforward" / "no_learning_placebo", RUNS / "walkforward" / "no_learning"):
        if (path / "fills.parquet").exists() and (path / "situations.json").exists() and not args.replay:
            fills, situations = load_base(path)
            print(f"loaded base from {path}: {len(fills)} fills, {len(situations)} situations", flush=True)
            break
    if fills is None or not situations or args.replay:
        print(f"building no-learning base with situations ({START}–{END})", flush=True)
        fills, situations = capture_situations(START, END)
    fills = fills.copy()
    fills["as_of"] = pd.to_datetime(fills.as_of)
    situations = {(t, pd.Timestamp(a)): s for (t, a), s in situations.items()}
    frame = run_audit(fills, situations, tables)
    frame.to_csv(RESULTS / "gate_placebo.csv", index=False)
    md, comps = summarise(frame)
    comps.to_csv(RESULTS / "gate_placebo_pairs.csv", index=False)
    (RESULTS / "gate_placebo.md").write_text(md)
    print(md, flush=True)


if __name__ == "__main__":
    main()
