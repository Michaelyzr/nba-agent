"""Simulated users with and without the Coach agent on seeded League slates (docs/preregistration_coach.md).

    python -m evaluation.coach_sim                 # 20 slates x 10 decisions, test period, M4 forecaster
    python -m evaluation.coach_sim --reuse         # reuse the snapshots cached in runs/coach_sim/

This is a simulation of compliance, not a user study: rule-based personas stand in
for biased users and follow the Coach's nudge with probability c. Each persona plays
the same slates without the Coach and with it (c = 0.5 and 1.0), with the same
compliance draws, so differences are paired. Outcomes are judged by the market:
closing-line value (CLV), P&L after fees and fees paid. CIs resample slates, and
for P&L and CLV also game-days (evaluation/stats.py).

Writes evaluation/results/coach_sim.md, coach_sim.csv, coach_sim_decisions.csv and coach_sim.png.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from agents import league
from agents.coach import CHASE_MOVE, LONG_SHOT, paper_trade
from agents.coach_agent import advise
from evaluation.ablations import markdown
from evaluation.stats import REPS, SEED, day_table, paired

RESULTS = Path(__file__).resolve().parent / "results"
CACHE = Path(__file__).resolve().parent.parent / "runs" / "coach_sim"
N_SLATES, N_DECISIONS, PERIOD = 20, 10, "test"
STAKE, CAUTION_STAKE = 20.0, 10.0
COMPLIANCE = (0.5, 1.0)
CHASE_MIN, UNDERDOG_MAX, BIG_GAP = 0.01, 0.40, 0.10
SLATE_COLS = ["trades", "clv_sum", "clv_dollars", "pnl", "fees"]


def raw_gap(side):
    return side["p_model"] - side["breakeven"]


def chaser(s):
    side = max(s["sides"].values(), key=lambda x: x["move"], default=None)
    return side["team"] if side is not None and side["move"] >= CHASE_MIN else None


def long_shot_lover(s):
    side = min(s["sides"].values(), key=lambda x: x["ask"], default=None)
    return side["team"] if side is not None and side["ask"] <= UNDERDOG_MAX else None


def overtrader(s):
    side = max(s["sides"].values(), key=raw_gap, default=None)
    return side["team"] if side is not None else None


def cautious(s):
    side = max(s["sides"].values(), key=raw_gap, default=None)
    return side["team"] if side is not None and raw_gap(side) >= BIG_GAP else None


PERSONAS = {"price chaser": chaser, "long-shot lover": long_shot_lover, "overtrader": overtrader, "cautious": cautious}


def slate_names(n=N_SLATES):
    return [f"coach-sim-{i:02d}" for i in range(n)]


def build_slates(rp, forecaster, n_slates=N_SLATES, n_decisions=N_DECISIONS, period=PERIOD, names=None):
    """Each slate's decisions with the as-of Coach snapshot (the same one a League player sees)."""
    slates = []
    for name in slate_names(n_slates):
        lg = league.create(name, rp, period, n_decisions)
        snaps = [league.decision_info(forecaster, rp, lg, i, names)["snapshot"] for i in range(len(lg["slate"]))]
        slates.append({"name": name, "snapshots": snaps})
        print(f"  {name}: {len(snaps)} decisions")
    return slates


def draw(persona: str, slate: int, idx: int, seed: int = SEED) -> float:
    """The compliance draw u, shared by every arm so the arms are paired."""
    return float(np.random.default_rng([seed, list(PERSONAS).index(persona), slate, idx]).random())


def _history_entry(s, team):
    if team is None:
        return {"choice": "pass", "team": None}
    side = s["sides"][team]
    return {"choice": "trade", "team": team, "move": side["move"], "gap": side["gap"], "price": side["ask"],
            "chased": side["move"] >= CHASE_MOVE, "long_shot": side["ask"] <= LONG_SHOT}


def play(rp, slates, persona: str, compliance=None, seed: int = SEED, advise_fn=advise) -> list:
    """One persona over every slate; compliance None is the arm without the Coach (the nudge is still recorded)."""
    rule, rows = PERSONAS[persona], []
    arm = "without" if compliance is None else f"with c={compliance:g}"
    for k, slate in enumerate(slates):
        history = []
        for i, s in enumerate(slate["snapshots"]):
            team = rule(s)
            a = advise_fn(s, team, history) if team is not None else None
            nudge = a["nudge"] if a else None
            u = draw(persona, k, i, seed)
            complied = compliance is not None and nudge in ("pass", "caution") and u < compliance
            stake = STAKE
            if complied and nudge == "pass":
                team = None
            elif complied:
                stake = CAUTION_STAKE
            row = {"persona": persona, "arm": arm, "slate": k, "index": i, "game_id": s["game_id"],
                   "as_of": pd.Timestamp(s["as_of"]), "intended": rule(s), "nudge": nudge, "complied": complied,
                   "lesson": a["lesson"]["id"] if a else None, "checks": ",".join(x["check"] for x in a["steps"]) if a else "",
                   "team": team, "stake": stake if team else 0.0, "filled": False}
            if team is not None:
                t = paper_trade(rp, league._game(rp, s["game_id"]), pd.Timestamp(s["as_of"]), s["sides"][team], stake, s)
                if t["filled"]:
                    row.update({k2: t[k2] for k2 in ("filled", "price", "contracts", "fee", "close_price", "clv",
                                                      "clv_dollars", "pnl", "won")})
            history.append(_history_entry(s, team))
            rows.append(row)
    return rows


def run_all(rp, slates, seed: int = SEED) -> pd.DataFrame:
    rows = []
    for persona in PERSONAS:
        for c in (None, *COMPLIANCE):
            rows += play(rp, slates, persona, c, seed)
    d = pd.DataFrame(rows)
    for col in ("price", "contracts", "fee", "close_price", "clv", "clv_dollars", "pnl"):
        d[col] = pd.to_numeric(d.get(col), errors="coerce")
    return d


def slate_table(d: pd.DataFrame, n_slates: int) -> pd.DataFrame:
    f = d[d.filled]
    t = pd.DataFrame({"trades": f.groupby("slate").size(), "clv_sum": f.groupby("slate").clv.sum(),
                      "clv_dollars": f.groupby("slate").clv_dollars.sum(), "pnl": f.groupby("slate").pnl.sum(),
                      "fees": f.groupby("slate").fee.sum()})
    return t.reindex(range(n_slates), fill_value=0.0).fillna(0.0)[SLATE_COLS].astype(float)


def _metrics(sums):
    trades, clv_sum, clv_dollars, pnl, fees = np.moveaxis(np.asarray(sums, float), -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_clv = np.where(trades > 0, clv_sum / np.where(trades > 0, trades, 1), 0.0)
    return {"pnl": pnl, "clv_dollars": clv_dollars, "mean_clv": mean_clv, "trades": trades, "fees": fees}


def slate_paired(a: pd.DataFrame, b: pd.DataFrame, reps=REPS, seed=SEED) -> dict:
    """a - b on the same slates: estimate and 95% percentile CI per metric, resampling slates."""
    idx = np.random.default_rng(seed).integers(0, len(a), size=(reps, len(a)))
    pa, pb = _metrics(a.to_numpy().sum(0)), _metrics(b.to_numpy().sum(0))
    ba, bb = _metrics(a.to_numpy()[idx].sum(1)), _metrics(b.to_numpy()[idx].sum(1))
    out = {}
    for m in pa:
        draws = ba[m] - bb[m]
        out[m] = (float(pa[m] - pb[m]), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
    return out


def day_paired(a: pd.DataFrame, b: pd.DataFrame, games: pd.DataFrame, days) -> pd.DataFrame:
    def table(d):
        f = d[d.filled][["game_id", "clv", "contracts", "pnl", "price"]]
        return day_table(f, games, days)
    return paired(table(a), table(b))


def nudge_accuracy(d: pd.DataFrame, n_slates: int, reps=REPS, seed=SEED) -> pd.DataFrame:
    """Without the Coach: mean CLV per contract of trades by the nudge the Coach would have given."""
    f = d[(d.arm == "without") & d.filled]
    rows = []
    for persona, g in [("all personas", f), *f.groupby("persona")]:
        row = {"persona": persona}
        for n in ("pass", "caution", "none"):
            x = g[g.nudge == n]
            row[f"{n} trades"] = len(x)
            row[f"{n} mean CLV"] = float(x.clv.mean()) if len(x) else np.nan
        p = g[g.nudge == "pass"].groupby("slate").clv.agg(["sum", "count"]).reindex(range(n_slates), fill_value=0)
        o = g[g.nudge == "none"].groupby("slate").clv.agg(["sum", "count"]).reindex(range(n_slates), fill_value=0)
        idx = np.random.default_rng(seed).integers(0, n_slates, size=(reps, n_slates))
        with np.errstate(invalid="ignore", divide="ignore"):
            draws = (p["sum"].to_numpy()[idx].sum(1) / p["count"].to_numpy()[idx].sum(1)
                     - o["sum"].to_numpy()[idx].sum(1) / o["count"].to_numpy()[idx].sum(1))
        draws = draws[~np.isnan(draws)]
        row["pass - none"] = row["pass mean CLV"] - row["none mean CLV"]
        row["ci_low"], row["ci_high"] = ((float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))
                                         if len(draws) else (np.nan, np.nan))
        row["pass trades with CLV < 0"] = float((g[g.nudge == "pass"].clv < 0).mean()) if row["pass trades"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(d: pd.DataFrame, games: pd.DataFrame, n_slates: int) -> pd.DataFrame:
    days = games.loc[games.game_id.isin(d.game_id), "date"]
    rows = []
    for persona in PERSONAS:
        base = d[(d.persona == persona) & (d.arm == "without")]
        base_t = slate_table(base, n_slates)
        for arm in ["without", *[f"with c={c:g}" for c in COMPLIANCE]]:
            x = d[(d.persona == persona) & (d.arm == arm)]
            t = slate_table(x, n_slates)
            m = _metrics(t.to_numpy().sum(0))
            row = {"persona": persona, "arm": arm, "decisions": len(x), "trades": int(m["trades"]),
                   "pnl": float(m["pnl"]), "clv_dollars": float(m["clv_dollars"]),
                   "mean_clv": float(x[x.filled].clv.mean()) if x.filled.any() else np.nan,
                   "fees": float(m["fees"]), "worst_slate_pnl": float(t.pnl.min()),
                   "nudged_pass": int((x.nudge == "pass").sum()), "nudged_caution": int((x.nudge == "caution").sum()),
                   "complied": int(x.complied.sum())}
            if arm != "without":
                sp = slate_paired(t, base_t)
                dp = day_paired(x, base, games, days)
                for metric, (est, lo, hi) in sp.items():
                    row[f"d_{metric}"], row[f"d_{metric}_lo"], row[f"d_{metric}_hi"] = est, lo, hi
                for metric in ("pnl", "clv_dollars", "mean_clv"):
                    row[f"d_{metric}_day_lo"] = float(dp.loc[metric, "ci_low"])
                    row[f"d_{metric}_day_hi"] = float(dp.loc[metric, "ci_high"])
                row["d_worst_slate_pnl"] = float(t.pnl.min() - base_t.pnl.min())
            rows.append(row)
    rows.append({"persona": "never trade", "arm": "reference", "decisions": n_slates * N_DECISIONS, "trades": 0,
                 "pnl": 0.0, "clv_dollars": 0.0, "mean_clv": np.nan, "fees": 0.0, "worst_slate_pnl": 0.0})
    return pd.DataFrame(rows)


def _money(x):
    return "-" if pd.isna(x) else f"{'+' if x > 0 else '−' if x < 0 else ''}${abs(x):,.2f}"


def _ci(est, lo, hi, fmt=_money):
    return "-" if pd.isna(est) else f"{fmt(est)} [{fmt(lo)}, {fmt(hi)}]"


def _cents(x):
    return "-" if pd.isna(x) else f"{x * 100:+.2f}¢"


def plot(summary: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summary[summary.arm.str.startswith("with")]
    personas = list(PERSONAS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = (("d_pnl", "P&L change (with − without), $", 1), ("fees", "fees saved (without − with), $", -1),
              ("d_clv_dollars", "CLV $ change (with − without)", 1))
    for ax, (metric, title, sign) in zip(axes, panels):
        for j, c in enumerate(COMPLIANCE):
            r = s[s.arm == f"with c={c:g}"].set_index("persona").loc[personas]
            col = "d_fees" if metric == "fees" else metric
            est, lo, hi = sign * r[col], sign * r[f"{col}_lo"], sign * r[f"{col}_hi"]
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            x = np.arange(len(personas)) + (j - 0.5) * 0.38
            ax.bar(x, est, 0.38, yerr=[est - lo, hi - est], capsize=3, label=f"compliance {c:g}",
                   color=["#9ecae1", "#3182bd"][j])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(personas)), personas, rotation=15)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle("Simulated users with the Coach agent: 20 slates x 10 test-period decisions "
                 "(95% slate-bootstrap CIs; a simulation of compliance, not a user study)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(summary: pd.DataFrame, acc: pd.DataFrame, n_slates: int, n_decisions: int) -> str:
    main = summary.assign(**{"P&L": summary.pnl.map(_money), "CLV $": summary.clv_dollars.map(_money),
                             "mean CLV": summary.mean_clv.map(_cents), "fees": summary.fees.map(_money),
                             "worst slate": summary.worst_slate_pnl.map(_money)})
    t1 = main[["persona", "arm", "trades", "P&L", "CLV $", "mean CLV", "fees", "worst slate",
               "nudged_pass", "nudged_caution", "complied"]]
    w = summary[summary.arm.str.startswith("with")]
    t2 = pd.DataFrame({
        "persona": w.persona, "arm": w.arm,
        "Δ P&L [slate CI]": [_ci(*r) for r in zip(w.d_pnl, w.d_pnl_lo, w.d_pnl_hi)],
        "Δ P&L day CI": [f"[{_money(a)}, {_money(b)}]" for a, b in zip(w.d_pnl_day_lo, w.d_pnl_day_hi)],
        "Δ CLV $ [slate CI]": [_ci(*r) for r in zip(w.d_clv_dollars, w.d_clv_dollars_lo, w.d_clv_dollars_hi)],
        "Δ CLV $ day CI": [f"[{_money(a)}, {_money(b)}]" for a, b in zip(w.d_clv_dollars_day_lo, w.d_clv_dollars_day_hi)],
        "Δ mean CLV [slate CI]": [_ci(*r, fmt=_cents) for r in zip(w.d_mean_clv, w.d_mean_clv_lo, w.d_mean_clv_hi)],
        "Δ fees [slate CI]": [_ci(*r) for r in zip(w.d_fees, w.d_fees_lo, w.d_fees_hi)],
        "Δ trades [slate CI]": [_ci(*r, fmt=lambda v: f"{v:+.0f}") for r in zip(w.d_trades, w.d_trades_lo, w.d_trades_hi)],
        "Δ worst slate": w.d_worst_slate_pnl.map(_money)})
    t3 = acc.assign(**{c: acc[c].map(_cents) for c in ("pass mean CLV", "caution mean CLV", "none mean CLV",
                                                      "pass - none", "ci_low", "ci_high")})
    t3["pass trades with CLV < 0"] = acc["pass trades with CLV < 0"].map(lambda v: "-" if pd.isna(v) else f"{v:.0%}")
    return (f"# Coach agent: simulated users with and without the Coach\n\n"
            f"**This is a simulation of compliance, not a user study.** Rule-based personas stand in for biased "
            f"users and follow the Coach's nudge with probability c. Design, personas and hypotheses were "
            f"pre-registered in `docs/preregistration_coach.md` before this run.\n\n"
            f"{n_slates} seeded League slates x {n_decisions} decisions from the test period (as-of information "
            f"only), $20 per trade (halved to $10 when a caution nudge is followed). Fills at the recorded ask plus "
            f"fee; CLV against the last price before tip-off. Paired: same persona, slates and compliance draws in "
            f"every arm. CIs are 95% percentile bootstraps over slates (2000 reps); 'day CI' resamples game-days "
            f"with `evaluation/stats.py`.\n\n"
            f"## Totals per persona and arm\n\n{markdown(t1)}\n"
            f"## Paired differences: with Coach − without Coach\n\n{markdown(t2)}\n"
            f"## Was the nudge right? (without-Coach trades, labelled by the nudge the Coach would have given)\n\n"
            f"Mean CLV per contract; 'pass − none' with a slate-bootstrap CI.\n\n{markdown(t3)}\n"
            f"## Limitations\n\n"
            f"- Personas are fixed rules; they do not learn across slates, and real users may ignore or "
            f"over-follow the Coach. Compliance is assumed, not measured.\n"
            f"- The Coach and the overtrader and cautious personas read the same M4 snapshot, so some agreement "
            f"is by construction; outcomes are therefore judged by market CLV, not by the Coach's own estimate.\n"
            f"- P&L over 200 decisions is mostly luck; CLV and fees are the more reliable outcomes.\n"
            f"- Slates may share games; the day-clustered CI accounts for shared nights.\n"
            f"- The optional LLM wording is not evaluated: it never changes the nudge.\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--slates", type=int, default=N_SLATES)
    ap.add_argument("--decisions", type=int, default=N_DECISIONS)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args(argv)
    import replay
    from data_sources import FROZEN
    tables = replay.load_tables(FROZEN)
    rp = replay.Replay(tables, lambda *a: [])
    cache = CACHE / f"slates_{args.slates}x{args.decisions}.pkl"
    if args.reuse and cache.exists():
        slates = pickle.loads(cache.read_bytes())
    else:
        from forecast.api import Forecaster
        slates = build_slates(rp, Forecaster.load(), args.slates, args.decisions)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(slates))
    d = run_all(rp, slates)
    summary = summarise(d, tables["games"], len(slates))
    acc = nudge_accuracy(d, len(slates))
    RESULTS.mkdir(exist_ok=True)
    d.to_csv(RESULTS / "coach_sim_decisions.csv", index=False)
    summary.to_csv(RESULTS / "coach_sim.csv", index=False)
    acc.to_csv(RESULTS / "coach_sim_nudges.csv", index=False)
    (RESULTS / "coach_sim.md").write_text(report(summary, acc, len(slates), args.decisions))
    plot(summary, RESULTS / "coach_sim.png")
    print((RESULTS / "coach_sim.md").read_text())


if __name__ == "__main__":
    main()
