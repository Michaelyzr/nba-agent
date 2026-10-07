"""PNG of the thin multi-agent night graph for the presentation deck.

    python -m evaluation.orchestrator_diagram
    # -> evaluation/results/orchestrator.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RESULTS = Path(__file__).resolve().parent / "results"
NODES = [
    ("user", "User\n(game night)", 0.08, 0.55, "#7f7f7f"),
    ("supervisor", "Supervisor\n(LangGraph)", 0.30, 0.55, "#1f77b4"),
    ("pregame", "Pregame\nnews + fair odds", 0.55, 0.82, "#2ca02c"),
    ("trader", "Trader\nMarketAgent", 0.55, 0.55, "#ff7f0e"),
    ("coach", "Coach\nexplain / teach", 0.55, 0.28, "#9467bd"),
    ("briefs", "Briefs\n4 channels", 0.80, 0.55, "#8c564b"),
]
EDGES = [("user", "supervisor"), ("supervisor", "pregame"), ("pregame", "supervisor"),
         ("supervisor", "trader"), ("trader", "supervisor"), ("supervisor", "coach"),
         ("coach", "supervisor"), ("supervisor", "briefs"), ("briefs", "supervisor")]


def draw(path: Path = RESULTS / "orchestrator.png"):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = {}
    for key, label, x, y, color in NODES:
        w, h = 0.16, 0.14
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.012",
                             facecolor=color, edgecolor="black", alpha=0.85, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        boxes[key] = (x, y)
    for a, b in EDGES:
        x1, y1 = boxes[a]
        x2, y2 = boxes[b]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     color="#333333", lw=1.2, connectionstyle="arc3,rad=0.05",
                                     shrinkA=28, shrinkB=28))
    ax.text(0.5, 0.05, "One shared as-of time (tip − 60 min). Typed handoffs. Failures degrade, never look ahead.",
            ha="center", fontsize=9, style="italic")
    ax.set_title("E. Night orchestrator: pregame → trader → coach → briefs", fontsize=12, pad=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    draw()
