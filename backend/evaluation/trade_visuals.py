"""Figures that explain how the agent trades: the decision flow, one real trade step by step, and the funnel.

    python -m evaluation.trade_visuals                         # uses runs/results/test-full and data/frozen
    python -m evaluation.trade_visuals --decision 401810565-2  # pick another filled decision for the example

Writes trade_flow.png, trade_example.png and trade_funnel.png to reports/results/.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from agents.graph import ANCHOR_LEAD, DEFAULT_MIN_EDGE, DEFAULT_STAKE, fee_per_contract
from replay import RUNS, load_tables
from nba_agent_paths import FROZEN, RESULTS

INK, RED, GREEN, BLUE, ORANGE, GREY = "#44506a", "#c92a2a", "#2b8a3e", "#3b5bdb", "#e8590c", "#868e96"


# ---------------- 1. decision flow ----------------

def _box(ax, x, y, w, h, title, sub, fc):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=fc, ec=INK, lw=1.2, zorder=2))
    ax.text(x, y + (0.13 if sub else 0), title, ha="center", va="center", fontsize=10.5, weight="bold", zorder=3)
    if sub:
        ax.text(x, y - 0.17, sub, ha="center", va="center", fontsize=7.8, color="#333", zorder=3)


def _diamond(ax, x, y, w, h, text, fc="#fff3bf"):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)], closed=True,
                         fc=fc, ec=INK, lw=1.2, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.6, weight="bold", zorder=3, linespacing=1.1)


def _arrow(ax, a, b, label="", color=INK, rad=0.0, lpos=(0, 0)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=1))
    if label:
        ax.text((a[0] + b[0]) / 2 + lpos[0], (a[1] + b[1]) / 2 + lpos[1], label, fontsize=8.2, color=color,
                style="italic", ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.4), zorder=4)


def _elbow(ax, points, label="", color=INK, lpos=None):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.3, zorder=1, solid_joinstyle="round")
    _arrow(ax, points[-2], points[-1], color=color)
    if label:
        x, y = lpos
        ax.text(x, y, label, fontsize=8.2, color=color, style="italic", ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.4), zorder=4)


def flow(path: Path):
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(8.5, 9.7, "How the agent decides and places a trade (one game, one decision time)", ha="center",
            fontsize=16, weight="bold")
    W, H, DW, DH = 3.3, 0.75, 2.6, 1.0
    PASS = "#f1f3f5"

    # column 1: what do we believe?
    x1 = 2.2
    ax.text(x1, 9.15, "1. What do we believe?", ha="center", fontsize=12, weight="bold", color=BLUE)
    steps1 = [("Decision time", "each news item within 6 h of tip, and tip − 60 min", "#e8eefc"),
              ("Who is out?", "official statuses published so far", "#e8eefc"),
              ("M4: P(win) twice", "without vs with the missing players", "#e8eefc"),
              ("News shift", "after − before (e.g. −8.7 points)", "#e8eefc"),
              ("Anchor", "market mid 24 h before tip", "#e8eefc"),
              ("Our estimate", "anchor + news shift", "#d0ebff")]
    ys1 = [8.4, 7.2, 6.0, 4.8, 3.6, 2.4]
    for (t, s, c), y in zip(steps1, ys1):
        _box(ax, x1, y, W, H, t, s, c)
    for a, b in zip(ys1, ys1[1:]):
        _arrow(ax, (x1, a - H / 2), (x1, b + H / 2))
    ax.text(x1, 1.35, "The market is the base rate;\nthe model only adds what the news changed.",
            ha="center", fontsize=8.8, style="italic", color=BLUE)

    # column 2: is there value?
    x2 = 7.0
    ax.text(x2, 9.15, "2. Is there value after costs?", ha="center", fontsize=12, weight="bold", color=ORANGE)
    _box(ax, x2, 8.4, W + 0.4, H, "Price each side", "YES at the ask, NO at 1 − bid, + fee 7%·p·(1−p)", "#fff4e6")
    _box(ax, x2, 7.2, W + 0.4, H, "Gap after fees", "our P(side) − price − fee; keep the bigger side", "#fff4e6")
    _diamond(ax, x2, 5.85, DW, DH, f"gap >\n{DEFAULT_MIN_EDGE * 100:.0f} points?")
    _diamond(ax, x2, 4.35, DW, DH, "a notebook\nrule matches?")
    _diamond(ax, x2, 2.85, DW, DH, "already hold\nthis game?")
    _box(ax, x2, 1.5, W, H, "Propose order", f"\\${DEFAULT_STAKE:.0f} (a rule may scale it)", "#ffe8cc")
    _elbow(ax, [(x1 + W / 2, 2.4), (4.5, 2.4), (4.5, 8.4), (x2 - (W + 0.4) / 2, 8.4)])
    _arrow(ax, (x2, 8.4 - H / 2), (x2, 7.2 + H / 2))
    _arrow(ax, (x2, 7.2 - H / 2), (x2, 5.85 + DH / 2))
    _arrow(ax, (x2, 5.85 - DH / 2), (x2, 4.35 + DH / 2), "yes", GREEN, lpos=(0.25, 0))
    _arrow(ax, (x2, 4.35 - DH / 2), (x2, 2.85 + DH / 2), "no / raise the bar", GREEN, lpos=(0.75, 0))
    _arrow(ax, (x2, 2.85 - DH / 2), (x2, 1.5 + H / 2), "no", GREEN, lpos=(0.2, 0))
    px = 9.6
    _box(ax, px, 4.35, 1.5, 3.4, "PASS", "", PASS)
    ax.text(px, 3.55, "brief only:\n'already priced in'\nor 'rule r023'\nor 'one position\nper game'",
            ha="center", va="center", fontsize=7.6, color="#333")
    for y, lab in ((5.85, "no"), (4.35, "skip"), (2.85, "yes")):
        _arrow(ax, (x2 + DW / 2, y), (px - 0.75, y), lab, RED, lpos=(0, 0.14))

    # column 3: is it safe, and what happens?
    x3 = 13.2
    ax.text(x3, 9.15, "3. Is it safe? What happens?", ha="center", fontsize=12, weight="bold", color=RED)
    _diamond(ax, x3, 8.25, DW + 0.3, DH, "checks pass?\ncitations, numbers,\nwording")
    _diamond(ax, x3, 6.7, DW + 0.3, DH, "risk limits?\n\\$50 order · \\$100 game\n\\$300 day · before tip")
    _box(ax, x3, 5.35, W, H, "Confirm", "retail orders need a human", "#fde8e8")
    _box(ax, x3, 4.15, W + 0.6, H, "Fill", "at the recorded ask + fee; quote ≤ 5 min old;\n≤ 10% of last hour's volume",
         "#e3f5e6")
    _box(ax, x3, 2.9, W + 0.6, H, "Settle", "closing-line value = price at tip − our price;\nP&L after the final buzzer",
         "#e3f5e6")
    _box(ax, x3, 1.6, W + 0.6, H, "Nightly review", "worst trades → one rule → gate on earlier days → notebook",
         "#fdf1e3")
    _elbow(ax, [(x2 + W / 2, 1.5), (10.85, 1.5), (10.85, 8.25), (x3 - (DW + 0.3) / 2, 8.25)])
    _arrow(ax, (x3, 8.25 - DH / 2), (x3, 6.7 + DH / 2), "pass", GREEN, lpos=(0.3, 0))
    _arrow(ax, (x3, 6.7 - DH / 2), (x3, 5.35 + H / 2), "approved", GREEN, lpos=(0.45, 0))
    for a, b in ((5.35, 4.15), (4.15, 2.9), (2.9, 1.6)):
        _arrow(ax, (x3, a - H / 2), (x3, b + H / 2))
    bx = 16.1
    _box(ax, bx, 7.45, 1.4, 2.0, "BLOCKED", "", "#ffe3e3")
    ax.text(bx, 7.05, "reason\nlogged", ha="center", va="center", fontsize=7.6)
    _arrow(ax, (x3 + (DW + 0.3) / 2, 8.25), (bx - 0.7, 8.0), "fail twice\n(1 retry)", RED, lpos=(0, 0.2))
    _arrow(ax, (x3 + (DW + 0.3) / 2, 6.7), (bx - 0.7, 6.9), "over", RED, lpos=(0, 0.13))
    _elbow(ax, [(x3, 1.6 - H / 2), (x3, 0.55), (5.15, 0.55), (5.15, 4.35), (x2 - DW / 2, 4.35)],
           "new rules apply from the next day", "#a08000", lpos=(9.2, 0.55))

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------- 2. one real trade ----------------

def example(decision_id: str, run: Path, path: Path):
    tables = load_tables()
    d = pd.read_parquet(run / "decisions.parquet").set_index("decision_id").loc[decision_id]
    f = pd.read_parquet(run / "fills.parquet").set_index("decision_id").loc[decision_id]
    trace = next(t for t in map(json.loads, (run / "trace.jsonl").read_text().splitlines())
                 if t.get("phase") == "decide" and t["game_id"] == d.game_id and pd.Timestamp(t["as_of"]) == d.as_of)
    g = tables["games"].set_index("game_id").loc[d.game_id]
    team = d.market_ticker.rsplit("-", 1)[-1]
    p = tables["prices"]
    p = p[p.market_ticker == d.market_ticker].sort_values("ts").assign(mid=lambda x: (x.bid + x.ask) / 2)
    tip = pd.Timestamp(g.tip_time)
    anchor_row = p[p.ts <= tip - ANCHOR_LEAD]
    anchor = float((anchor_row.iloc[-1] if len(anchor_row) else p.iloc[0])[["bid", "ask"]].mean())
    shift = float(re.search(r"news shift ([+-][\d.]+) points", d.reason).group(1)) / 100
    est = anchor + shift
    yes = d.side == "yes"
    p_side, price, fee = (est if yes else 1 - est), float(f.price), fee_per_contract(float(f.price))
    breakeven, gap = price + fee, (est if yes else 1 - est) - price - fee
    close = float(f.close_price)
    news = tables["news"]
    news = news[(news.game_id == d.game_id) & (news.published_at <= d.as_of)]
    players = pd.read_parquet(FROZEN / "players.parquet")
    names = dict(zip(players.player_id, players.player_name))
    who = [names.get(pid, str(pid)) for pid in news.player_id]
    side_word = f"{team} wins" if yes else f"{team} does NOT win"

    fig, (a, b) = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1.15, 1]})
    window = p[(p.ts >= tip - pd.Timedelta(hours=26)) & (p.ts <= tip)]
    a.plot(window.ts, window.mid * 100, color=INK, lw=1.4, label=f"{team} win price (mid)")
    a.axvline(tip - ANCHOR_LEAD, color=BLUE, ls=":", lw=1.2)
    a.scatter([tip - ANCHOR_LEAD], [anchor * 100], color=BLUE, zorder=5, s=50)
    a.annotate(f"anchor: {anchor:.0%}\n(24 h before tip)", (tip - ANCHOR_LEAD, anchor * 100), xytext=(12, 18),
               textcoords="offset points", color=BLUE, fontsize=9)
    a.axvline(d.as_of, color=ORANGE, ls="--", lw=1.2)
    a.annotate(f"news {pd.Timestamp(d.as_of):%H:%M} UTC:\n{len(who)} {team} players out", (d.as_of, 95 if yes else 15),
               xytext=(-150, 0), textcoords="offset points", color=ORANGE, fontsize=9,
               arrowprops=dict(arrowstyle="->", color=ORANGE))
    fill_y = price if yes else 1 - price
    a.scatter([d.as_of], [fill_y * 100], color=GREEN, marker="v" if not yes else "^", s=110, zorder=6)
    a.annotate(f"agent buys {d.side.upper()} at {price:.0%}\n(= {team} at {fill_y:.0%})", (d.as_of, fill_y * 100),
               xytext=(-175, -40), textcoords="offset points", color=GREEN, fontsize=9,
               arrowprops=dict(arrowstyle="->", color=GREEN))
    close_y = close if yes else 1 - close
    a.scatter([tip], [close_y * 100], color=RED, s=60, zorder=6)
    a.annotate(f"close at tip: {close_y:.1%}", (tip, close_y * 100), xytext=(-110, -30), textcoords="offset points",
               color=RED, fontsize=9, arrowprops=dict(arrowstyle="->", color=RED))
    a.scatter([d.as_of], [est * 100], color=BLUE, marker="*", s=160, zorder=6)
    a.annotate(f"our estimate: {est:.0%}\n= {anchor:.0%} {shift * 100:+.1f} pts", (d.as_of, est * 100),
               xytext=(15, -10), textcoords="offset points", color=BLUE, fontsize=9)
    a.set(ylim=(0, 100), ylabel=f"{team} win probability / price (%)",
          title=f"{g.away_team} @ {g.home_team}, {g.date}: the price over the last 26 hours")
    a.legend(loc="upper left", fontsize=9)
    fig.autofmt_xdate()

    b.set_xlim(0, 100)
    b.set_ylim(0, 10)
    b.axis("off")
    b.set_title(f"The decision in numbers: buying '{side_word}'", fontsize=12)
    rows = [
        ("Who is out", ", ".join(who) if who else "nobody", INK),
        ("M4 news shift", f"{team} {shift * 100:+.1f} points", INK),
        ("Our estimate", f"{team} {est:.1%}  →  P({side_word}) = {p_side:.1%}", BLUE),
        ("Cost", f"price {price:.0%} + fee {fee * 100:.1f}¢  =  break-even {breakeven:.1%}", ORANGE),
        ("Gap after fees", f"{gap * 100:+.1f} points  >  {DEFAULT_MIN_EDGE * 100:.0f}-point threshold  →  trade", GREEN),
        ("Order", f"{int(f.contracts)} contracts at {price:.0%} (\\${DEFAULT_STAKE:.0f} stake), fee \\${f.fee:.2f}", INK),
        ("Closing price", f"{close:.1%}  →  closing-line value {f.clv * 100:+.1f}¢ per contract "
                          f"(\\${f.clv * f.contracts:+.2f})", RED),
        ("Result", f"final {g.away_team} {g.away_pts:.0f}, {g.home_team} {g.home_pts:.0f}  →  "
                   f"P&L after fees \\${f.pnl:+.2f}", GREEN if f.pnl > 0 else RED),
    ]
    for i, (k, v, c) in enumerate(rows):
        y = 9.5 - i * 0.78
        b.text(0, y, k, fontsize=10, weight="bold", va="center")
        b.text(27, y, v, fontsize=9.6, va="center", color=c, wrap=True)
    y0 = 0.9
    b.plot([5, 95], [y0, y0], color=GREY, lw=2)
    for v in range(0, 101, 10):
        x = 5 + v * 0.9
        b.plot([x, x], [y0 - 0.08, y0 + 0.08], color=GREY)
        b.text(x, y0 - 0.38, f"{v}", ha="center", fontsize=7.5, color=GREY)
    b.text(50, y0 - 0.85, f"probability that {side_word} (%)", ha="center", fontsize=8.5, color=GREY)
    for v, lab, c, dy in ((price, "price", ORANGE, 0.45), (breakeven, "break-even", ORANGE, 0.95),
                          (close, "close", RED, 1.45), (p_side, "our estimate", BLUE, 1.95)):
        x = 5 + v * 90
        b.plot([x, x], [y0, y0 + dy - 0.15], color=c, lw=1.5)
        b.text(x, y0 + dy, f"{lab} {v:.0%}", ha="center", fontsize=8.5, color=c, weight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


# ---------------- 3. funnel ----------------

def funnel(run: Path, path: Path):
    traces = [t for t in map(json.loads, (run / "trace.jsonl").read_text().splitlines()) if t.get("phase") == "decide"]
    fills = pd.read_parquet(run / "fills.parquet")
    no_quote = no_edge = rule_or_held = blocked = 0
    for t in traces:
        detail = next((s["detail"] for s in t["trace"] if s["step"] == "analyse"), "")
        gaps = [float(x) for x in re.findall(r"gap ([+-][\d.]+)", detail)]
        if t["status"] == "sent":
            continue
        if t["status"] == "blocked":
            blocked += 1
        elif not gaps:
            no_quote += 1
        elif max(gaps) <= DEFAULT_MIN_EDGE:
            no_edge += 1
        else:
            rule_or_held += 1
    stages = [("Decision points\n(news or tip − 60 min)", len(traces)),
              ("with a fresh price", len(traces) - no_quote),
              (f"gap after fees > {DEFAULT_MIN_EDGE * 100:.0f} points", len(traces) - no_quote - no_edge),
              ("not skipped by a rule,\nnot already holding the game", len(traces) - no_quote - no_edge - rule_or_held),
              ("passed checks and risk:\nfilled", len(fills)),
              ("beat or matched the closing price", int((fills.clv >= 0).sum())),
              ("won", int((fills.pnl > 0).sum()))]
    fig, ax = plt.subplots(figsize=(11, 6))
    top = stages[0][1]
    for i, (label, n) in enumerate(stages):
        w = n / top
        color = BLUE if i < 4 else (GREEN if i == 4 else "#69db7c")
        ax.barh(-i, w, left=(1 - w) / 2, color=color, height=0.75)
        if w > 0.12:
            ax.text(0.5, -i, f"{n}", ha="center", va="center", color="white", weight="bold", fontsize=11)
        else:
            ax.text((1 + w) / 2 + 0.015, -i, f"{n}  ({n / top:.0%} of decision points)" if i < 5 else
                    f"{n}  ({n / len(fills):.0%} of trades)", ha="left", va="center", color=INK, weight="bold",
                    fontsize=10.5)
        ax.text(-0.02, -i, label, ha="right", va="center", fontsize=9.5)
    ax.set_xlim(-0.55, 1.02)
    ax.axis("off")
    ax.set_title(f"Test period (1 Feb – 12 Apr 2026): from {top} decision points to {len(fills)} trades", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=RUNS / "results" / "test-full")
    ap.add_argument("--decision", default="401810565-2")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    flow(args.out / "trade_flow.png")
    example(args.decision, args.run, args.out / "trade_example.png")
    for label, n in funnel(args.run, args.out / "trade_funnel.png"):
        print(f"{n:5d}  {label.replace(chr(10), ' ')}")
    print(f"wrote trade_flow.png, trade_example.png, trade_funnel.png to {args.out}")


if __name__ == "__main__":
    main()
