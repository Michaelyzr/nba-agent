"""How the M4 win model does against the market on every test game, as a table and a four-panel figure.

    python -m evaluation.m4_report                     # data/frozen, test period, writes reports/results/m4_*

Per game: M4 before the inactive list (tip - 60 min), M4 after it (tip - 1 min,
official absences known), the market mid at both moments, and the result.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agents.graph import OUT_STATUSES, load_source
from replay import AsOf, Replay

from nba_agent_paths import RESULTS
PREDICTORS = {"home_rate": "Always 55% home", "m4_before": "M4, before inactive list",
              "m4_after": "M4, absences known", "market_before": "Market, 1 h before tip",
              "market_tip": "Market at tip"}


def game_table(tables, forecaster, start, end) -> pd.DataFrame:
    rp = Replay(tables, lambda *a: [])
    games = tables["games"]
    games = games[(games.date >= start) & (games.date <= end)].dropna(subset=["home_pts"])
    markets = tables["markets"][tables["markets"].kind == "game"]
    rows = []
    for g in games.itertuples():
        m = markets[(markets.game_id == g.game_id) & (markets.team == g.home_team)]
        if m.empty:
            continue
        ticker = m.market_ticker.iloc[0]
        before = AsOf(tables, pd.Timestamp(g.tip_time) - pd.Timedelta(minutes=60), rp.price_index)
        after = AsOf(tables, pd.Timestamp(g.tip_time) - pd.Timedelta(minutes=1), rp.price_index)
        q_before, q_tip = before.quote(ticker), AsOf(tables, g.tip_time, rp.price_index).quote(ticker)
        if q_before is None or q_tip is None:
            continue
        news = after.news(g.game_id)
        out = news.loc[news.status.astype(str).str.lower().isin(OUT_STATUSES), "player_id"].tolist()
        rows.append({"game_id": g.game_id, "date": g.date, "matchup": f"{g.away_team} @ {g.home_team}",
                     "players_out": len(out), "home_rate": 0.55,
                     "m4_before": forecaster.win(before, g, [])["p_home"],
                     "m4_after": forecaster.win(after, g, out)["p_home"],
                     "market_before": float((q_before.bid + q_before.ask) / 2),
                     "market_tip": float((q_tip.bid + q_tip.ask) / 2),
                     "home_win": float(g.home_pts > g.away_pts)})
    return pd.DataFrame(rows)


def scores(t: pd.DataFrame) -> pd.DataFrame:
    y = t.home_win.to_numpy()
    out = []
    for col, label in PREDICTORS.items():
        p = np.clip(t[col].to_numpy(), 0.01, 0.99)
        out.append({"predictor": label, "games": len(t), "brier": float(np.mean((p - y) ** 2)),
                    "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                    "accuracy": float(np.mean((p > 0.5) == y))})
    return pd.DataFrame(out)


def disagreement(t: pd.DataFrame, cut=0.05) -> pd.DataFrame:
    """When M4 and the market at tip disagree by more than `cut`, how often did the home team actually win?"""
    d = t.m4_after - t.market_tip
    groups = {f"M4 higher on home by >{cut:.0%}": d > cut, "within ±5%": d.abs() <= cut,
              f"M4 lower on home by >{cut:.0%}": d < -cut}
    return pd.DataFrame([{"group": k, "games": int(v.sum()), "M4 said": t.m4_after[v].mean(),
                          "market said": t.market_tip[v].mean(), "home actually won": t.home_win[v].mean()}
                         for k, v in groups.items()])


def figure(t: pd.DataFrame, s: pd.DataFrame, dis: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    colors = {"m4_after": "#3b5bdb", "market_tip": "#e8590c", "m4_before": "#91a7ff"}

    a = ax[0, 0]
    bins = np.linspace(0, 1, 11)
    for col, label in (("m4_after", "M4, absences known"), ("market_tip", "Market at tip")):
        b = pd.cut(t[col], bins, include_lowest=True)
        g = t.groupby(b, observed=True).agg(p=(col, "mean"), y=("home_win", "mean"), n=("home_win", "size"))
        g = g[g.n >= 5]
        a.plot(g.p, g.y, "o-", color=colors[col], label=label)
    a.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    a.set(xlabel="predicted home win probability", ylabel="actual home win rate", xlim=(0, 1), ylim=(0, 1),
          title="Calibration (bins with 5+ games)")
    a.legend(loc="upper left")

    a = ax[0, 1]
    order = s.sort_values("brier", ascending=False)
    bar_colors = ["#adb5bd", "#91a7ff", "#3b5bdb", "#ffa94d", "#e8590c"]
    bars = a.barh(order.predictor, order.brier, color=[bar_colors[list(PREDICTORS.values()).index(p)]
                                                       for p in order.predictor])
    for bar, (_, r) in zip(bars, order.iterrows()):
        a.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
               f"{r.brier:.3f}  ({r.accuracy:.0%} right)", va="center", fontsize=9)
    a.set(xlabel="Brier score (lower is better)", xlim=(0.14, order.brier.max() + 0.03),
          title=f"Accuracy on {len(t)} test games")

    a = ax[1, 0]
    x = np.arange(len(dis))
    w = 0.27
    a.bar(x - w, dis["M4 said"], w, color=colors["m4_after"], label="M4 said")
    a.bar(x, dis["market said"], w, color=colors["market_tip"], label="Market said")
    a.bar(x + w, dis["home actually won"], w, color="#2b8a3e", label="Home actually won")
    a.set_xticks(x, [f"{g}\n({n} games)" for g, n in zip(dis.group, dis.games)], fontsize=9)
    a.set(ylabel="home win probability / rate", ylim=(0, 1), title="When M4 disagrees with the market, who is right?")
    a.legend(fontsize=9)

    a = ax[1, 1]
    t = t.sort_values("date").reset_index(drop=True)
    for col, label in (("m4_after", "M4, absences known"), ("market_tip", "Market at tip")):
        err = (t[col] - t.home_win) ** 2
        a.plot(pd.to_datetime(t.date), err.rolling(60, min_periods=30).mean(), color=colors[col], label=label)
    a.set(ylabel="Brier score, rolling 60 games", title="Over the test period")
    a.legend()
    fig.autofmt_xdate()

    fig.suptitle("M4 win model vs the Kalshi market, test period", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    from forecast.api import Forecaster

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "frozen", "sample"], default="frozen")
    ap.add_argument("--start", default="2026-02-01")
    ap.add_argument("--end", default="2026-04-12")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    tables = load_source(args.source, args.start, args.end)
    t = game_table(tables, Forecaster.load(), args.start, args.end)
    s, dis = scores(t), disagreement(t)
    t.to_csv(args.out / "m4_games.csv", index=False)
    s.to_csv(args.out / "m4_vs_market.csv", index=False)
    dis.to_csv(args.out / "m4_disagreement.csv", index=False)
    figure(t, s, dis, args.out / "m4_vs_market.png")
    print(s.round(4).to_string(index=False))
    print(dis.round(3).to_string(index=False))
    print(f"wrote {args.out}/m4_vs_market.png")


if __name__ == "__main__":
    main()
