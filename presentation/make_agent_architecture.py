"""Draw presentation/figures/agent_architecture.png: decide loop, learn loop, people, guardrails, grader.

    python presentation/make_agent_architecture.py

Mirrors agents/graph.py (decide and review graphs), agents/tool_agent.py (LLM analyst + sceptic)
and agents/orchestrator.py (Coach and League consume the briefs).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures" / "agent_architecture.png"

NAVY, BLUE, ORANGE, GREEN, RED, GOLD = "#1E293B", "#2563EB", "#EA580C", "#16A34A", "#DC2626", "#B45309"
FILL = {"decide": "#DBEAFE", "llm": "#EDE9FE", "guard": "#FEE2E2", "learn": "#FFEDD5",
        "people": "#DCFCE7", "end": "#E2E8F0"}
EDGE = {"decide": BLUE, "llm": "#7C3AED", "guard": RED, "learn": ORANGE, "people": GREEN, "end": NAVY}

W, H = 17.2, 6.9
BW, GAP = 1.72, 0.17


def box(ax, x, y, w, h, title, sub, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=FILL[kind], ec=EDGE[kind], lw=2.4 if kind == "guard" else 1.6))
    ax.text(x + w / 2, y + h - 0.27, title, ha="center", va="center", fontsize=12, weight="bold", color=NAVY)
    ax.text(x + w / 2, y + (h - 0.45) / 2, sub, ha="center", va="center", fontsize=10, color="#334155",
            linespacing=1.25)


def arrow(ax, a, b, color=NAVY, rad=0.0, ls="-", lw=1.8, label=None, lxy=None, ha="center"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        ax.text(*lxy, label, fontsize=10.5, color=color, style="italic", ha=ha, va="center")


def main():
    plt.rcParams["text.parse_math"] = False
    fig, ax = plt.subplots(figsize=(W, H), dpi=200)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---- decide loop
    ax.text(0.2, 6.7, "DECIDE  ·  at every news time and at tip − 60 min, the agent chooses when to act, "
            "what to look at and whether to trade", fontsize=13.5, weight="bold", color=BLUE, va="center")
    y, h = 3.95, 1.75
    decide = [
        ("Trigger", "news arrives\nor clock\n(tip − 60 min)", "decide"),
        ("As-of tools", "prices · news\nrotation\nM4 win · M6 move", "decide"),
        ("Estimate", "market anchor\n+ M4 news shift,\nor LLM analyst", "llm"),
        ("Edge", "gap after\nspread + fee\n≥ 4¢ ?", "decide"),
        ("Sceptic", "LLM agent:\n“already priced\nin?” can veto", "llm"),
        ("Code checks", "re-checks every\nnumber in code;\none retry", "guard"),
        ("Risk", "$50 order,\n$100 game, $300\nday; kill switch", "guard"),
        ("Confirm", "a human\nconfirms\nretail orders", "decide"),
        ("Order or pass", "fill at ask + fee;\n~90% pass,\nwith the reason", "end"),
    ]
    xs = [0.2 + i * (BW + GAP) for i in range(len(decide))]
    for x, (t, s, k) in zip(xs, decide):
        box(ax, x, y, BW, h, t, s, k)
    for i in range(len(xs) - 1):
        arrow(ax, (xs[i] + BW, y + h / 2), (xs[i + 1], y + h / 2))

    # guardrail tags above the decide row
    for i, t in [(1, "GUARD:\nno look-ahead"), (2, "GUARD:\nnumbers from code"),
                 (6, "GUARD:\nLLM can't override")]:
        ax.text(xs[i] + BW / 2 + 0.0, 6.0, t, ha="center", va="center", fontsize=10,
                color=RED, weight="bold")

    # ---- learn loop, right to left under the decide row
    ly, lh, lw = 1.2, 1.55, 1.8
    learn = [
        ("Settle at close", "fills settled;\nclosing price\nrecorded", 15.2),
        ("CLV grade", "did we beat\nthe closing\nprice?", 13.1),
        ("Reviewer", "worst slice,\nproposes\none rule", 11.0),
        ("Gate", "must help on\nheld-out days;\nplacebo-audited", 8.9),
        ("Notebook", "kept rules:\n“when X, pass”;\nexpire in 45 d", 6.8),
    ]
    for t, s, x in learn:
        box(ax, x, ly, lw, lh, t, s, "learn")
    for (_, _, xa), (_, _, xb) in zip(learn, learn[1:]):
        arrow(ax, (xa, ly + lh / 2), (xb + lw, ly + lh / 2))
    ax.text(14.9, 3.15, "LEARN  ·  once a day, graded by the market", fontsize=13.5, weight="bold",
            color=ORANGE, va="center", ha="right")
    arrow(ax, (xs[8] + BW / 2, y), (15.2 + lw / 2, ly + lh), label="next day", lxy=(16.25, 3.35), ha="left")
    arrow(ax, (6.8 + 0.4, ly + lh), (xs[2] + BW * 0.8, y), color=GOLD, ls="--", rad=-0.2, lw=2.0,
          label="rules feed back into\nestimate / decide", lxy=(6.9, 3.35), ha="left")
    ax.text(9.8, 0.98, "rejected rules are dropped", fontsize=10, color="#64748B", ha="center", style="italic")

    # ---- people
    box(ax, 0.2, ly, 3.95, lh, "People: Coach and League",
        "Coach explains each game with the\nagent's analysis and break-even;\n"
        "League ranks users by CLV, not profit", "people")
    arrow(ax, (xs[2] + 0.3, y), (3.0, ly + lh), color=GREEN, rad=0.15, label="briefs +\nanalysis",
          lxy=(3.25, 3.3), ha="left")

    # legend
    for i, (k, lab) in enumerate([("decide", "deterministic step"), ("llm", "LLM step"),
                                  ("guard", "code guardrail")]):
        yy = 2.5 - i * 0.42
        ax.add_patch(FancyBboxPatch((4.45, yy - 0.13), 0.38, 0.26, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc=FILL[k], ec=EDGE[k], lw=1.5))
        ax.text(4.95, yy, lab, fontsize=10.5, va="center", color=NAVY)

    # ---- market as grader
    ax.add_patch(FancyBboxPatch((0.2, 0.12), W - 0.4, 0.62, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#FEF3C7", ec=GOLD, lw=1.6))
    ax.text(W / 2, 0.43, "MARKET AS GRADER:  the closing price grades every box  —  estimates (Brier), "
            "trades (CLV), rules (held-out gate), people (League rank)", fontsize=11.5, weight="bold",
            color=GOLD, ha="center", va="center")

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
