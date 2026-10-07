"""Cumulative P&L after fees over time, from the saved replay runs.

    python -m evaluation.cumulative_pnl     # writes evaluation/results/cumulative_pnl.{png,csv}

Left: the 1 Feb - 12 Apr 2026 test period (runs/results, runs/m6). Right: the
walk-forward season, 1 Nov - 12 Apr regular season (runs/walkforward), so the
end points match walkforward_summary.md. Each fill counts on its game's date
(settlement). Never trading is $0 by construction. Thin dotted lines show the
cumulative closing-line value of the same fills, as in headline_clv.png.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
RUNS = ROOT / "runs"

PANELS = {
    "Test period, 1 Feb – 12 Apr 2026": ("2026-02-01", "2026-04-12", {
        "Full agent (anchor + learning)": "results/test-full",
        "Agent, no learning": "results/test-no-learning",
        "Raw model, no anchor": "results/test-no-anchor",
        "M6 agent (no trades)": "m6/test-m6-learning",
    }),
    "Walk-forward season, 1 Nov – 12 Apr": ("2025-11-01", "2026-04-12", {
        "Full agent (anchor + learning)": "walkforward/full",
        "Agent, no learning": "walkforward/no_learning",
        "Raw model, no anchor": "walkforward/raw",
    }),
}
COLORS = {"Full agent (anchor + learning)": "#2563EB", "Agent, no learning": "#EA580C",
          "Raw model, no anchor": "#DC2626", "M6 agent (no trades)": "#16A34A"}


def cumulative(run: str, games: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d")
    fills = pd.read_parquet(RUNS / run / "fills.parquet")
    if fills.empty:
        daily = pd.DataFrame(0.0, index=days, columns=["pnl", "clv"])
    else:
        f = fills.dropna(subset=["outcome"]).merge(games[["game_id", "date"]], on="game_id")
        f = f[(f.date >= start) & (f.date <= end)]
        f = f.assign(clv_usd=f.clv * f.contracts)
        daily = (f.groupby("date").agg(pnl=("pnl", "sum"), clv=("clv_usd", "sum"))
                 .reindex(days, fill_value=0.0))
    out = daily.cumsum()
    out.index = pd.to_datetime(out.index)
    trades = 0 if fills.empty else int(((fills.merge(games[["game_id", "date"]], on="game_id").date
                                         .between(start, end))).sum())
    return out.assign(trades=trades)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 13})
    games = pd.read_parquet(ROOT / "data" / "frozen" / "games.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    rows = []
    for ax, (title, (start, end, runs)) in zip(axes, PANELS.items()):
        ax.axhline(0, color="black", lw=1.8, ls="--", label="Never trade ($0)", zorder=5)
        for label, run in runs.items():
            c = cumulative(run, games, start, end)
            end_pnl = c.pnl.iloc[-1]
            money = f"{'−' if end_pnl < -0.5 else ''}${abs(end_pnl):,.0f}"
            ax.step(c.index, c.pnl, where="post", color=COLORS[label], lw=4 if c.trades.iloc[0] == 0 else 2,
                    label=f"{label}: {money} ({c.trades.iloc[0]} trades)")
            ax.step(c.index, c.clv, where="post", color=COLORS[label], lw=1, ls=":", alpha=0.8)
            rows.append({"panel": title, "setup": label, "trades": int(c.trades.iloc[0]),
                         "pnl_after_fees": round(float(c.pnl.iloc[-1]), 2),
                         "clv_usd": round(float(c.clv.iloc[-1]), 2),
                         "low_point": round(float(c.pnl.min()), 2)})
        ax.set_title(title)
        ax.legend(loc="lower left", fontsize=11, framealpha=0.95)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0].set_ylabel("cumulative P&L after fees ($)")
    fig.suptitle("Money over time, $20 orders: solid = P&L after fees, dotted = closing-line value",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS / "cumulative_pnl.png", dpi=150)
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "cumulative_pnl.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
