"""Draw the whole product, from public data to users, as one slide-ready PNG.

    python -m evaluation.product_diagram          # writes evaluation/results/product_workflow.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RESULTS = Path(__file__).resolve().parent / "results"
INK = "#44506a"
LANES = [
    (2.05, "1. Public data", "#495057", "#f1f3f5"),
    (6.15, "2. Models (trained once)", "#3b5bdb", "#e8eefc"),
    (10.25, "3. Replay (no future data)", "#0b7285", "#e3fafc"),
    (14.35, "4. Agent (LangGraph)", "#d9480f", "#fff4e6"),
    (18.45, "5. Outputs and users", "#2b8a3e", "#ebfbee"),
]
BOXES = {
    "espn": (0, 7.85, "ESPN", "games, box scores, inactive lists"),
    "kalshi": (0, 6.15, "Kalshi", "game-winner markets, prices, settlements"),
    "frozen": (0, 4.05, "Frozen tables", "games · player_games · news\nmarkets · prices · settlements"),
    "history": (1, 7.85, "Leakage-safe history", "only games that tipped off\nbefore this one"),
    "models": (1, 6.0, "M1 – M4", "M1 baselines  ·  M2 GRU (points/minutes)\nM3 P(plays)  ·  M4 P(win | who is out)"),
    "m5": (1, 4.05, "M5 Forecaster", "one interface; P(win) for trades,\nplayer tables for briefs"),
    "asof": (2, 7.85, "As-of view", "news, prices, finished games\nthat were public at that moment"),
    "times": (2, 6.15, "Decision times", "each news item within 6 h of tip,\nand once at tip − 60 min"),
    "fills": (2, 4.05, "Fills and settlement", "ask + fee, ≤ 10% of volume\nCLV vs price at tip; P&L"),
    "decide": (3, 7.85, "Decide", "investigate → M4 forecast → analyse\n(anchor + news shift) → propose / pass\n→ checks → risk → confirm"),
    "review": (3, 6.0, "Review (nightly)", "settle → worst CLV slice → one rule\n→ gate on earlier days"),
    "notebook": (3, 4.05, "Notebook", "versioned rules, 45-day expiry\nused by Decide from the next day"),
    "briefs": (4, 8.15, "Channel briefs", "platform · media · team · retail"),
    "orders": (4, 6.75, "Orders + audit", "paper orders, every step logged"),
    "coach": (4, 5.4, "Coach", "explain, lessons, paper trades"),
    "eval": (4, 4.05, "Evaluation", "ablations · M4 vs market · policy"),
}


def draw(path: Path):
    fig, ax = plt.subplots(figsize=(21, 10.2))
    ax.set_xlim(0, 20.5)
    ax.set_ylim(1.15, 10.15)
    ax.axis("off")
    ax.text(10.25, 9.85, "Market-graded NBA agent: the whole product", ha="center", fontsize=18, weight="bold")
    w, h = 3.5, 1.35
    for x, title, color, fill in LANES:
        ax.add_patch(FancyBboxPatch((x - 1.9, 3.15), 3.8, 6.15, boxstyle="round,pad=0.02,rounding_size=0.22",
                                    fc=fill, ec=color, lw=1.5, ls="--", zorder=0))
        ax.text(x, 9.0, title, ha="center", fontsize=13, weight="bold", color=color)
    pos = {}
    for name, (lane, y, title, lines) in BOXES.items():
        x, color = LANES[lane][0], LANES[lane][2]
        hh = 1.55 if lines.count("\n") >= 2 else h
        pos[name] = (x, y, hh)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - hh / 2), w, hh, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=color, lw=1.5, zorder=2))
        ax.text(x, y + hh / 2 - 0.28, title, ha="center", va="center", fontsize=12, weight="bold", zorder=3)
        ax.text(x, y - 0.15, lines, ha="center", va="center", fontsize=8.6, color="#333", zorder=3, linespacing=1.2)

    def arrow(a, b, color=INK):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15, lw=1.5, color=color,
                                     connectionstyle="arc3,rad=0", shrinkA=1, shrinkB=1, zorder=1))

    def elbow(points, color=INK):
        ax.plot(*zip(*points[:-1]), color=color, lw=1.5, zorder=1, solid_joinstyle="round")
        arrow(points[-2], points[-1], color)

    for a, b in (("espn", "frozen"), ("kalshi", "frozen"), ("history", "models"), ("models", "m5"),
                 ("asof", "times"), ("times", "fills"), ("decide", "review"), ("review", "notebook")):
        (xa, ya, ha), (xb, yb, hb) = pos[a], pos[b]
        arrow((xa, ya - ha / 2), (xb, yb + hb / 2))
    for a, b in (("frozen", "history"), ("frozen", "asof"), ("m5", "decide"), ("times", "decide"),
                 ("fills", "review"), ("decide", "briefs"), ("decide", "orders"), ("decide", "coach"),
                 ("fills", "eval")):
        (xa, ya, _), (xb, yb, _) = pos[a], pos[b]
        arrow((xa + w / 2, ya), (xb - w / 2, yb))
    xn, yn, hn = pos["notebook"]
    xd, yd, hd = pos["decide"]
    elbow([(xn + 0.7, yn + hn / 2), (xn + 0.7, yd), (xd + 0.7, yd - hd / 2)], "#a08000")
    ax.text(xn + 1.15, (yn + yd) / 2, "rules", fontsize=8.5, color="#a08000", style="italic", rotation=90,
            va="center")

    ax.text(10.25, 2.45, "Build:  python -m data_sources.espn  ·  data_sources.kalshi download"
            "     →     python -m forecast.train\n"
            "Run:    python -m agents.graph   ·   python -m evaluation.ablations"
            "   ·   streamlit run app.py",
            ha="center", va="center", fontsize=10.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.55", fc="#f8f9fa", ec="#adb5bd"))
    ax.text(10.25, 1.5, "Guardrails: public data only  ·  no future information  ·  checks + risk caps "
            "($50 / $100 / $300)  ·  rules gated on earlier days  ·  paper money  ·  LLM optional",
            ha="center", fontsize=10, color="#c92a2a", style="italic")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS / "product_workflow.png")
    args = ap.parse_args()
    draw(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
