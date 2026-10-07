"""'Graded by the market' tab of app.py: every decision, setup and learned rule scored by closing-line value.

Reads committed result files in evaluation/results/ (and an optional run's fills); a missing file degrades to a
message. Kept apart from app.py so it can be smoke-tested without models or replay data.
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RESULTS = Path(__file__).resolve().parent / "evaluation" / "results"

EXPLAINER = (
    "**The market is the grader.** Every decision is scored by **closing-line value (CLV)**: the side's mid price at "
    "tip-off (the close) minus the price we paid. The close is the market's last, best-informed estimate before the "
    "game starts, so buying below it means we bought information before the market priced it in. Unlike one night's "
    "profit, CLV is not decided by who wins the game, which makes it a far less noisy measure of skill. Fills are "
    "at the **ask** (we pay the spread) plus **Kalshi's fee**, so a trade has to beat the close by more than those "
    "costs before it makes money. Profit and loss (P&L) after fees at settlement is shown next to CLV. Learning "
    "(review → rules → gate) is only accepted if the market grades it better."
)

PUNCHLINE = ("Every upgrade that \"wins\" does so by **trading less**. All learned rules only remove trades (skip a "
             "slice of markets or raise the minimum edge), and random placebo rules pass the same gate about as often "
             "as the reviewer's. The market is hard to beat after fees, which is why the product is education, not "
             "a trading bot.")

PENDING = "pending (API quota)"


def read_csv(name: str, results: Path = RESULTS):
    path = results / name
    return pd.read_csv(path) if path.exists() else None


def _num(x):
    return x is not None and not pd.isna(x)


def ci_text(est, lo, hi, kind):
    """'-0.35¢ [-0.65, +0.00]' for kind 'cents' (input in dollars per contract), '-$35 [-60, -12]' for 'dollars'."""
    if not _num(est):
        return "–"
    k = 100 if kind == "cents" else 1
    head = f"{est * k:+.2f}¢" if kind == "cents" else f"{'−' if est < 0 else '+'}${abs(est):,.0f}"
    if not (_num(lo) and _num(hi)):
        return head
    spec = "+.2f" if kind == "cents" else "+,.0f"
    return f"{head} [{lo * k:{spec}}, {hi * k:{spec}}]"


def _setup_row(group, source, name, r, period="Test, 1 Feb – 12 Apr 2026"):
    return {"group": group, "setup": name, "period": period, "trades": int(r.trades) if _num(r.trades) else 0,
            "mean CLV per contract [95% CI]": ci_text(r.mean_clv, r.mean_clv_lo, r.mean_clv_hi, "cents"),
            "CLV $ [95% CI]": ci_text(r.clv_dollars, r.clv_dollars_lo, r.clv_dollars_hi, "dollars"),
            "P&L after fees [95% CI]": ci_text(r.pnl, r.pnl_lo, r.pnl_hi, "dollars"), "source": source}


def scoreboard(results: Path = RESULTS):
    """(table of every setup graded by the market with 95% day-clustered bootstrap CIs, missing file names)."""
    rows, missing = [], []
    ga = read_csv("gate_audit.csv", results)
    if ga is None:
        missing.append("gate_audit.csv")
    else:
        names = {"split_nolearn": "Deterministic agent, no learning", "split_full": "Deterministic agent + learning "
                 "(split gate)", "legacy_full": "Deterministic agent + learning (legacy gate)", "never": "Never trade"}
        for r in ga[ga.period == "test"].itertuples():
            if r.setup_id in names:
                rows.append(_setup_row("Deterministic agent", "gate_audit.csv", names[r.setup_id], r))
    llm = read_csv("llm_agent.csv", results)
    if llm is None:
        missing.append("llm_agent.csv")
    else:
        period = "Test, 40% subsample of decision points"
        for r in llm[(llm.kind == "setup") & (llm.window == "test")].itertuples():
            rows.append(_setup_row("LLM comparison", "llm_agent.csv", r.label, r, period))
    rows.append({"group": "LLM comparison", "setup": "C/D. LLM tool agent (+ sceptic)", "period": "Test",
                 "trades": None, "mean CLV per contract [95% CI]": PENDING, "CLV $ [95% CI]": PENDING,
                 "P&L after fees [95% CI]": PENDING, "source": "llm_agent.md"})
    ho = read_csv("holdout.csv", results)
    if ho is None:
        missing.append("holdout.csv")
    else:
        for r in ho.itertuples():
            rows.append(_setup_row("Play-off holdout", "holdout.csv", r.setup, r, r.window))
    table = pd.DataFrame(rows)
    if len(table):
        table["trades"] = table.trades.astype("Int64")
    return table, missing


def coach_board(results: Path = RESULTS):
    """Simulated biased users with and without the Coach's nudges; Δ CLV $ has a slate-bootstrap 95% CI."""
    cs = read_csv("coach_sim.csv", results)
    if cs is None:
        return None
    return pd.DataFrame({
        "persona": cs.persona, "arm": cs.arm, "trades": cs.trades,
        "CLV $": [ci_text(v, None, None, "dollars") for v in cs.clv_dollars],
        "P&L after fees": [ci_text(v, None, None, "dollars") for v in cs.pnl],
        "Δ CLV $ vs without Coach [95% CI]": [ci_text(d, lo, hi, "dollars") if _num(d) else "–"
                                              for d, lo, hi in zip(cs.d_clv_dollars, cs.d_clv_dollars_lo,
                                                                   cs.d_clv_dollars_hi)],
        "Δ trades": [f"{d:+.0f}" if _num(d) else "–" for d in cs.d_trades]})


def _rule_text(rule, action):
    try:
        when = {k: v for k, v in json.loads(rule).items() if k != "market_kind"}
    except (TypeError, ValueError):
        when = {}
    what = "skip" if action == "skip_market" else "raise the minimum edge on"
    cond = ", ".join(f"{k} = {v}" for k, v in when.items()) or "all game markets"
    return f"{what} trades where {cond}" if when else f"{what} all game markets"


def learning_loop(results: Path = RESULTS, mode: str = "split"):
    """Reviewer proposals at each review point, the gate's decision and the CLV change on gate days."""
    gp = read_csv("gate_placebo.csv", results)
    if gp is None:
        return None, None
    rev = gp[(gp["mode"] == mode) & (gp.kind == "reviewer")]
    loop = pd.DataFrame({
        "review day": rev.review_day, "proposed rule": [_rule_text(r, a) for r, a in zip(rev.rule, rev.action)],
        "trades it removes": rev.cases, "gate": rev.accepted.map({1: "accepted", 0: "rejected"}),
        "CLV $ on gate days, before → after": [f"{b:+.2f} → {a:+.2f}" for b, a in
                                               zip(rev.clv_dollars_before, rev.clv_dollars_after)],
        "Δ CLV $ (gate days)": rev.delta_clv_dollars.round(2),
        "Δ CLV $ (next 7 days)": rev.forward_clv_dollars.round(2)})
    rates = (gp.groupby(["mode", "kind"]).accepted.agg(["count", "sum"]).reset_index()
             .rename(columns={"count": "rules gated", "sum": "accepted"}))
    rates["pass rate"] = (rates.accepted / rates["rules gated"]).map("{:.0%}".format)
    return loop, rates


def grade_fills(fills: pd.DataFrame) -> pd.DataFrame:
    """One row per filled decision: entry (ask), fee, close (mid at tip), CLV and P&L after fees."""
    f = fills.reset_index(drop=True)
    outcome = f.outcome if "outcome" in f else pd.Series([None] * len(f))
    result = ["–" if pd.isna(o) else ("won" if (o == 1) == (s == "yes") else "lost") for o, s in zip(outcome, f.side)]
    return pd.DataFrame({
        "decision (UTC)": pd.to_datetime(f.as_of).dt.strftime("%H:%M"), "market": f.market_ticker,
        "side": f.side.str.upper(), "contracts": f.contracts, "entry (ask)": f.price.round(3),
        "fee $": f.fee.round(2), "close (mid at tip)": f.close_price.round(3),
        "CLV ¢": (f.clv * 100).round(2), "CLV $": (f.clv * f.contracts).round(2),
        "settled": result, "P&L after fees $": f.pnl.round(2)})


def path_chart(path: pd.DataFrame, fill, tip):
    """Side price from the decision to tip with the entry (ask paid) and the close marked."""
    entry = pd.DataFrame({"ts": [pd.Timestamp(fill.as_of)], "price": [fill.price], "mark": ["entry (ask paid)"]})
    close = pd.DataFrame({"ts": [pd.Timestamp(tip)], "price": [fill.close_price], "mark": ["close (mid at tip)"]})
    x = alt.X("ts:T", title="time (UTC)", scale=alt.Scale(type="utc"))
    line = alt.Chart(path).mark_line().encode(x=x, y=alt.Y("mid:Q", title=f"{fill.side.upper()} mid",
                                                           scale=alt.Scale(zero=False)))
    points = alt.Chart(pd.concat([entry, close])).mark_point(size=120, filled=True).encode(
        x="ts:T", y="price:Q", color=alt.Color("mark:N", title=None,
                                               scale=alt.Scale(range=["#d62728", "#2ca02c"])),
        tooltip=["mark", alt.Tooltip("price:Q", format=".3f")])
    return alt.layer(line, points).properties(height=220)


def render(fills=None, fills_source="", tip=None, prices=None, results: Path = RESULTS):
    """fills: graded fills for the selected game (or None); prices(ticker, start, end) -> DataFrame[ts, bid, ask]."""
    st.markdown(EXPLAINER)

    st.subheader("Every decision on this game, graded")
    if fills is None or len(fills) == 0:
        st.info("No filled decisions to grade for this game. Run the agent on the **Replayed night** tab, or pick a "
                "rule-notebook run whose saved fills cover this night." + (f" ({fills_source})" if fills_source else ""))
    else:
        st.caption(f"Graded fills from {fills_source}.")
        g = grade_fills(fills)
        st.dataframe(g, hide_index=True, width="stretch")
        a, b, c = st.columns(3)
        a.metric("Mean CLV per contract", f"{g['CLV ¢'].mean():+.2f}¢")
        b.metric("CLV $", f"${g['CLV $'].sum():+.2f}")
        c.metric("P&L after fees", f"${g['P&L after fees $'].sum():+.2f}", f"fees ${g['fee $'].sum():.2f}",
                 delta_color="off")
        if prices is not None and tip is not None:
            for fill in fills.itertuples():
                p = prices(fill.market_ticker, pd.Timestamp(fill.as_of) - pd.Timedelta(minutes=30), tip)
                if p is None or p.empty:
                    st.caption(f"No price path for {fill.market_ticker}.")
                    continue
                mid = (p.bid + p.ask) / 2
                path = p.assign(mid=mid if fill.side == "yes" else 1 - mid)
                st.altair_chart(path_chart(path, fill, tip), width="stretch")
                st.caption(f"{fill.side.upper()} {fill.market_ticker}: bought at {fill.price:.0%} plus "
                           f"${fill.fee:.2f} fee, closed at {fill.close_price:.1%} → CLV {fill.clv * 100:+.1f}¢.")

    st.subheader("Scoreboard: every setup graded by the market")
    board, missing = scoreboard(results)
    if len(board):
        st.dataframe(board, hide_index=True, width="stretch")
    st.caption("95% CIs: day-clustered bootstrap, 2000 replicates. Stakes are $20 per trade. Never trading scores "
               "exactly $0, and no setup's CLV $ interval lies above it. The LLM tool agent has no result yet: the "
               "API quota ran out, so no numbers are shown.")
    for name in missing:
        st.warning(f"Missing result file: evaluation/results/{name}")
    with st.expander("Coach: simulated biased users, with and without nudges"):
        cb = coach_board(results)
        if cb is None:
            st.warning("Missing result file: evaluation/results/coach_sim.csv")
        else:
            st.dataframe(cb, hide_index=True, width="stretch")
            st.caption("c = share of nudges followed. CIs: bootstrap over slates. CLV improves for every persona "
                       "mainly because they trade less.")
    money = read_csv("cumulative_pnl.csv", results)
    with st.expander("Money over time (the deck's cumulative P&L)"):
        if money is None:
            st.warning("Missing result file: evaluation/results/cumulative_pnl.csv")
        else:
            st.dataframe(money, hide_index=True, width="stretch")
            if (results / "cumulative_pnl.png").exists():
                st.image(str(results / "cumulative_pnl.png"))

    st.subheader("The learning loop, graded by the market")
    st.write("At each review point the reviewer proposes a rule from recent trades. The gate replays the rule on "
             "held-out gate days and accepts it only if the market grades it better there (CLV $ up).")
    loop, rates = learning_loop(results)
    if loop is None:
        st.warning("Missing result file: evaluation/results/gate_placebo.csv")
    else:
        st.dataframe(loop, hide_index=True, width="stretch")
        st.caption("Split gate. Δ CLV $ = CLV $ with the rule minus without it. Rules only remove trades, so a "
                   "rule 'helps' when the trades it drops had negative CLV.")
        with st.expander("Placebo check: random rules through the same gate"):
            st.dataframe(rates, hide_index=True, width="stretch")
    st.success(PUNCHLINE)
