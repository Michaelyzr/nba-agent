"""What data we use: row counts and date ranges of every frozen table, plus a timeline of the splits.

    python -m evaluation.data_overview     # writes evaluation/results/data_overview.csv and data_timeline.png

Every number is measured from data/frozen/*.parquet. Splits follow README: M1-M4 train on games
before 1 Feb 2026; agent development 1 Nov - 31 Jan; test 1 Feb - 12 Apr; walk-forward 1 Nov -
12 Apr with M4 retrained monthly; M6 trains before 15 Jan and validates on 15-31 Jan; play-in and
play-off games after 12 Apr are the holdout.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "data" / "frozen"
RESULTS = ROOT / "evaluation" / "results"


def load() -> dict:
    return {t: pd.read_parquet(FROZEN / f"{t}.parquet")
            for t in ["games", "markets", "settlements", "prices", "player_games", "players", "news"]}


def overview(t: dict) -> pd.DataFrame:
    g, m, p, n = t["games"], t["markets"], t["prices"], t["news"]
    priced = m[m.market_ticker.isin(p.market_ticker.unique())]
    priced_games = g[g.game_id.isin(priced.game_id)]
    news = n.merge(g[["game_id", "date"]], on="game_id")
    day = lambda s: pd.Timestamp(s).strftime("%Y-%m-%d")
    rows = [
        ("prices", len(p), day(p.ts.min()), day(p.ts.max()),
         f"{priced.market_ticker.nunique()} markets, {priced_games.game_id.nunique()} games"),
        ("markets", len(m), "", "", f"{m.game_id.nunique()} games, kind={'/'.join(m.kind.unique())}"),
        ("settlements", len(t["settlements"]), day(t["settlements"].settled_at.min()),
         day(t["settlements"].settled_at.max()), "outcome 0/1"),
        ("games", len(g), g.date.min(), g.date.max(), f"{g.season.nunique()} seasons"),
        ("player_games", len(t["player_games"]), "", "", f"{t['player_games'].player_id.nunique()} players"),
        ("players", len(t["players"]), "", "", ""),
        ("news", len(n), news.date.min(), news.date.max(),
         f"{news.game_id.nunique()} games; source={'/'.join(n.source.unique())}"),
    ]
    out = pd.DataFrame(rows, columns=["table", "rows", "first", "last", "note"])
    s26 = g[g.season == "2025-26"]
    splits = {
        "priced games: regular": int((priced_games.season_type == "regular").sum()),
        "priced games: play-in + play-offs": int((priced_games.season_type != "regular").sum()),
        "test games 1 Feb - 12 Apr": int(s26.date.between("2026-02-01", "2026-04-12").sum()),
        "holdout games after 12 Apr (priced)": int(priced_games.date.gt("2026-04-12").sum()),
        "news 2025-26 last date": news[news.game_id.isin(s26.game_id)].date.max(),
    }
    extra = pd.DataFrame([(k, v, "", "", "split") for k, v in splits.items()], columns=out.columns)
    return pd.concat([out, extra], ignore_index=True)


def timeline(t: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 13})
    g, p, n = t["games"], t["prices"], t["news"]
    priced = g[g.game_id.isin(t["markets"].game_id)]
    test_n = int(priced.date.between("2026-02-01", "2026-04-12").sum())
    hold_n = int(priced.date.gt("2026-04-12").sum())
    news_end = pd.Timestamp(n.merge(g[["game_id", "date"]], on="game_id").date.max())
    ts = lambda s: pd.Timestamp(s)
    lanes = [  # (label, [(start, end, colour, text)])
        ("Kalshi prices (1-min)", [(p.ts.min().tz_localize(None), p.ts.max().tz_localize(None), "#64748B",
                                     f"{len(p) / 1e6:.1f}M rows, {g[g.game_id.isin(t['markets'].game_id)].shape[0]:,} games")]),
        ("ESPN inactive lists", [(ts("2025-10-21"), news_end, "#94A3B8", f"news ends {news_end:%-d %b}")]),
        ("M1–M4 models", [(ts("2025-10-21"), ts("2026-01-31"), "#1E293B", "train (+ 2023-24, 2024-25)")]),
        ("Agent", [(ts("2025-11-01"), ts("2026-01-31"), "#93C5FD", "development"),
                   (ts("2026-02-01"), ts("2026-04-12"), "#2563EB", f"test: {test_n} games"),
                   (ts("2026-04-13"), ts("2026-06-14"), "#16A34A", f"holdout: {hold_n} games")]),
        ("Walk-forward", [(ts("2025-11-01"), ts("2026-04-12"), "#EA580C", "M4 retrained monthly")]),
        ("M6 (price-move net)", [(ts("2025-10-21"), ts("2026-01-14"), "#7C3AED", "train"),
                                 (ts("2026-01-15"), ts("2026-01-31"), "#C4B5FD", ""),
                                 (ts("2026-02-01"), ts("2026-04-12"), "#2563EB", "test"),
                                 (ts("2026-04-13"), ts("2026-06-14"), "#16A34A", "holdout")]),
    ]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    for i, (label, bars) in enumerate(lanes):
        y = len(lanes) - 1 - i
        for start, end, colour, text in bars:
            ax.barh(y, (end - start).days + 1, left=start, height=0.62, color=colour, edgecolor="white")
            if text:
                dark = colour in ("#1E293B", "#2563EB", "#7C3AED", "#16A34A", "#EA580C", "#64748B")
                ax.text(start + (end - start) / 2, y, text, ha="center", va="center", fontsize=12,
                        color="white" if dark else "#0F172A")
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([label for label, _ in reversed(lanes)])
    for day, text in [("2026-02-01", "1 Feb"), ("2026-04-13", "13 Apr")]:
        ax.axvline(ts(day), color="#0F172A", lw=1, ls="--", zorder=0)
        ax.text(ts(day), -1.15, text, ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.annotate("val 15–31 Jan", (ts("2026-01-23"), 0.33), (ts("2025-12-10"), -0.85), fontsize=11,
                arrowprops={"arrowstyle": "-", "color": "#475569"}, ha="center")
    ax.set_xlim(ts("2025-10-15"), ts("2026-06-20"))
    ax.set_ylim(-1.2, len(lanes) - 0.1)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("2025-26 season: what each part of the project sees", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    t = load()
    table = overview(t)
    table.to_csv(RESULTS / "data_overview.csv", index=False)
    print(table.to_string(index=False))
    timeline(t, RESULTS / "data_timeline.png")


if __name__ == "__main__":
    main()
