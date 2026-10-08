"""One-panel cumulative P&L after fees by game-day (test period 1 Feb – 12 Apr 2026).

    python -m evaluation.pnl_compare    # writes evaluation/results/pnl_compare.{png,csv}

Uses existing fills only (no re-runs, no LLM). Stake $20, fills at the ask + Kalshi fee.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
RUNS = ROOT / "runs"
START, END = "2026-02-01", "2026-04-12"

ARMS = [
    ("Raw M4, no agent", "m6/test-raw-m4", "#DC2626", "-"),
    ("Anchor agent, no learning", "m6/test-anchor-no-learning", "#EA580C", "-"),
    ("Split-gate learning", "gate_audit/test-split_full", "#2563EB", "-"),
    ("Adaptive edge, base 4¢", "adaptive_edge/adapt4", "#7C3AED", "-"),
    ("Never trade", None, "#6B7280", "--"),
]


def money(x):
    return f"−${abs(x):,.0f}" if round(x) < 0 else f"+${x:,.0f}" if round(x) > 0 else "$0"


def cumulative(run, dates):
    days = pd.date_range(START, END, freq="D")
    if run is None:
        return pd.Series(0.0, index=days), 0
    fills = pd.read_parquet(RUNS / run / "fills.parquet").assign(game_id=lambda d: d.game_id.astype(str))
    f = fills.merge(dates, on="game_id")
    f = f[(f.date >= START) & (f.date <= END)]
    s = f.groupby(pd.to_datetime(f.date)).pnl.sum().reindex(days, fill_value=0.0).cumsum()
    return s, len(f)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from data_sources import read_table

    games = read_table("games")
    dates = pd.DataFrame({"game_id": games.game_id.astype(str), "date": games.date.astype(str)})
    lines = [(label, *cumulative(run, dates), color, ls) for label, run, color, ls in ARMS]

    rows = [{"framework": label, "trades": n, "day": d.date().isoformat(), "cumulative_pnl": round(v, 2)}
            for label, cum, n, *_ in lines for d, v in cum.items()]
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "pnl_compare.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(0, color="k", lw=1, ls="--", zorder=1)
    for label, cum, n, color, ls in lines:
        cum = pd.concat([cum, pd.Series([cum.iloc[-1]], index=[cum.index[-1] + pd.Timedelta(days=1)])])
        ax.plot(cum.index, cum.values, color=color, ls=ls, lw=2.2,
                label=f"{label} ({n} trades): {money(cum.iloc[-1])}", drawstyle="steps-post")
    ax.set_title("Cumulative P&L after fees by game-day, 1 Feb – 12 Apr 2026\n"
                 "$20 stake, fills at the ask + Kalshi fee", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative P&L after fees")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: money(v)))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(pd.Timestamp(START), pd.Timestamp(END) + pd.Timedelta(days=1.5))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    fig.text(0.5, 0.01, "P&L is noisy (day-clustered 95% CIs ≈ ±$400–900). CLV is the primary metric.",
             ha="center", fontsize=9.5, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(RESULTS / "pnl_compare.png", dpi=150)

    for label, cum, n, *_ in lines:
        print(f"{label:30s} {n:4d} trades  {cum.iloc[-1]:+9.2f}")


if __name__ == "__main__":
    main()
