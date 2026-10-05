"""Draw the LangGraph agent workflow (agents/graph.py) as a slide-ready PNG.

    python -m evaluation.workflow_diagram          # writes evaluation/results/agent_workflow.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RESULTS = Path(__file__).resolve().parent / "results"
W, H = 2.3, 0.9
INK, RED, GREEN, GOLD = "#44506a", "#c92a2a", "#2b8a3e", "#a08000"
DECIDE, REVIEW, SAFETY, GOOD, BAD = "#e8eefc", "#fdf1e3", "#fde8e8", "#e3f5e6", "#f3e3e3"

NODES = {
    "trigger": (1.5, 7.3, "new news or tip − 60 min", DECIDE),
    "investigate": (4.1, 7.3, "news + statuses as of now", DECIDE),
    "forecast": (6.7, 7.3, "M2–M4: P(win), player lines", DECIDE),
    "analyse": (9.3, 7.3, "market anchor + news shift;\nmatch notebook rules", DECIDE),
    "propose": (11.9, 8.0, "order if gap after fees > edge", DECIDE),
    "no_action": (11.9, 6.6, "brief only", DECIDE),
    "checks": (14.5, 7.3, "citations, numbers, wording", SAFETY),
    "risk": (17.1, 7.3, "stake caps, daily loss limit", SAFETY),
    "blocked": (14.5, 5.5, "explain why, no order", SAFETY),
    "confirm": (17.1, 5.5, "retail orders need a human", SAFETY),
    "deliver": (15.8, 4.2, "channel briefs + audit log", DECIDE),
    "settle": (1.5, 1.75, "settle fills; CLV vs mid at tip", REVIEW),
    "review": (4.3, 1.75, "worst closing-line-value slice\n→ propose one rule", REVIEW),
    "gate": (7.1, 1.75, "backtest on earlier days only", REVIEW),
    "save_rule": (9.9, 2.45, "active from the next day", GOOD),
    "reject_rule": (9.9, 1.05, "may retry after 21 days", BAD),
}


def pt(name, side, dx=0.0):
    x, y = NODES[name][:2]
    return {"l": (x - W / 2, y), "r": (x + W / 2, y), "t": (x + dx, y + H / 2), "b": (x + dx, y - H / 2)}[side]


EDGES = [  # (from point, to point, label, arc, colour, label offset)
    (pt("trigger", "r"), pt("investigate", "l"), "", 0, INK, (0, 0)),
    (pt("investigate", "r"), pt("forecast", "l"), "", 0, INK, (0, 0)),
    (pt("forecast", "r"), pt("analyse", "l"), "", 0, INK, (0, 0)),
    (pt("analyse", "r"), pt("propose", "l"), "gap", 0, INK, (-0.15, 0.18)),
    (pt("analyse", "r"), pt("no_action", "l"), "priced in", 0, INK, (-0.1, -0.2)),
    (pt("propose", "r"), pt("checks", "l"), "", 0, INK, (0, 0)),
    (pt("no_action", "r"), pt("checks", "l"), "", 0, INK, (0, 0)),
    (pt("checks", "r"), pt("risk", "l"), "pass", 0, INK, (0, 0.17)),
    (pt("risk", "b"), pt("confirm", "t"), "approved", 0, INK, (0.45, 0)),
    (pt("risk", "b", -0.7), pt("blocked", "r"), "over limit", -0.25, INK, (0.25, 0.2)),
    (pt("checks", "b"), pt("blocked", "t"), "fails twice", 0, RED, (-0.5, 0)),
    (pt("blocked", "b"), pt("deliver", "l"), "", 0.25, INK, (0, 0)),
    (pt("confirm", "b"), pt("deliver", "r"), "", -0.25, INK, (0, 0)),
    (pt("checks", "t", 0.5), pt("investigate", "t"), "check fails → one retry with feedback", 0.25, RED, (0, 1.36)),
    (pt("settle", "r"), pt("review", "l"), "", 0, INK, (0, 0)),
    (pt("review", "r"), pt("gate", "l"), "", 0, INK, (0, 0)),
    (pt("gate", "r"), pt("save_rule", "l"), "helps", 0, GREEN, (-0.15, 0.18)),
    (pt("gate", "r"), pt("reject_rule", "l"), "does not help", 0, RED, (-0.1, -0.2)),
]


def box(ax, name, x, y, sub, color):
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=INK, lw=1.2, zorder=2))
    ax.text(x, y + 0.2, name, ha="center", va="center", fontsize=11.5, weight="bold", zorder=3)
    ax.text(x, y - 0.16, sub, ha="center", va="center", fontsize=7.8, color="#333", zorder=3, linespacing=1.1)


def arrow(ax, a, b, label="", rad=0.0, color=INK, loff=(0, 0), ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.3, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=1))
    if label:
        ax.text((a[0] + b[0]) / 2 + loff[0], (a[1] + b[1]) / 2 + loff[1], label, ha="center", va="center",
                fontsize=8.3, style="italic", color=color, zorder=4, bbox=dict(fc="white", ec="none", pad=0.6))


def lane(ax, x0, y0, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.2", fc="none", ec=color,
                                lw=1.6, ls="--", zorder=0))
    ax.text(x0 + 0.25, y0 + 0.28, title, fontsize=11.5, weight="bold", color=color, va="center")


def draw(path: Path):
    fig, ax = plt.subplots(figsize=(18, 9.9))
    ax.set_xlim(0, 18.6)
    ax.set_ylim(0, 9.9)
    ax.axis("off")
    ax.text(9.3, 9.68, "Market-graded NBA agent: LangGraph workflow", ha="center", fontsize=17, weight="bold")

    lane(ax, 0.1, 3.45, 18.4, 5.95, "DECIDE  ·  at every public news time within 6 h of tip, and at tip − 60 min",
         "#3b5bdb")
    lane(ax, 0.1, 0.2, 11.4, 2.95, "REVIEW  ·  once at the end of each day", "#d9822b")

    for name, (x, y, sub, color) in NODES.items():
        box(ax, name, x, y, sub, color)
    for a, b, label, rad, color, loff in EDGES:
        arrow(ax, a, b, label, rad, color, loff)

    nx, ny, nw, nh = 14.0, 1.75, 3.0, 1.6
    ax.add_patch(FancyBboxPatch((nx - nw / 2, ny - nh / 2), nw, nh, boxstyle="round,pad=0.02,rounding_size=0.3",
                                fc="#fff9db", ec=GOLD, lw=1.4, zorder=2))
    ax.text(nx, ny + 0.45, "Notebook", ha="center", va="center", fontsize=12, weight="bold", zorder=3)
    ax.text(nx, ny - 0.18, "versioned rules\nwhen {situation} → do {action}\nexpire after 45 days", ha="center",
            va="center", fontsize=8.2, zorder=3, linespacing=1.2)
    arrow(ax, pt("save_rule", "r"), (nx - nw / 2, ny + 0.45), "", 0, GREEN)
    arrow(ax, (nx - 0.6, ny + nh / 2), pt("analyse", "b"), "rules apply from valid_from", 0.15, GOLD, (0.4, -0.35),
          ls="--")

    ax.text(17.1, 1.75, "Inputs (public only)\n• ESPN box scores + inactive lists\n• Kalshi prices: fill at ask"
            " + fee,\n   capped by traded volume\n• Models: M2 GRU, M3 play, M4 win",
            fontsize=8.6, va="center", ha="center", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="#999"))

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS / "agent_workflow.png")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    draw(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
