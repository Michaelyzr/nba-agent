"""Offline full-game charts, using game-clock positions for both news measures."""
import json
from pathlib import Path

from agents.game_timeline import elapsed_seconds


def export_visuals(payload, output):
    output = Path(output)
    template = (Path(__file__).parent / "templates" / "loop_demo.html").read_text()
    from agents.match_view import match_view
    raw = [r for r in payload["steps"] if r["phase"] == "inplay"]
    def compact(s):
        fields = ('game_id', 'as_of', 'p_home', 'quote_state', 'score', 'market_quotes',
                  'model_trained', 'calibration_status', 'heldout_validation', 'errors', 'news_health', 'quote_quality', 'freshness')
        return {**{k: s.get(k) for k in fields}, 'report': {k: s.get('report', {}).get(k) for k in ('synthetic', 'news_effect')}}

    data_payload = {"home_team": payload["home_team"], "away_team": payload["away_team"],
                    "steps": [{"snapshot": compact(r["snapshot"]), "view": match_view(payload, raw[:i+1], r["snapshot"])} for i, r in enumerate(raw)]}
    data = json.dumps(data_payload, ensure_ascii=False, default=str, allow_nan=False).replace("<", "\\u003c")
    (output / "demo.html").write_text(template.replace("__LOOP_DATA__", data).replace("__ANALYSIS_SCRIPT__", (Path(__file__).parent / "templates" / "market_analysis.js").read_text()).replace("__DASHBOARD_SCRIPT__", (Path(__file__).parent / "templates" / "match_dashboard.js").read_text()))
    plot_sample(payload, output)


def plot_sample(payload, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    live = [r for r in payload["steps"] if r["phase"] == "inplay" and r["snapshot"].get("quote_state") != "final"]
    events = payload["timeline_events"]
    x = [elapsed_seconds(r["snapshot"]["score"]) / 60 for r in live]
    fig = plt.figure(figsize=(12, 10), layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=[3, 3, 1.9])
    axes = [fig.add_subplot(grid[i]) for i in range(2)]
    fig.suptitle(f"{payload['away_team']} at {payload['home_team']} | Q1 to Q4 news loop", fontsize=18)
    axes[0].plot(x, [r["snapshot"]["p_home"] * 100 for r in live], color="#087e70", linewidth=2.2, label=f"{payload['home_team']} model probability")
    axes[0].plot(x, [r["snapshot"]["report"]["reference_p_home"] * 100 for r in live], "--", color="#7c8a98", label="Without news (same score/clock)")
    axes[0].plot(x, [r["snapshot"]["market_quotes"]["home"]["mid"] * 100 for r in live], ":", color="#8b65ac", label="Synthetic market midpoint (not fill price)")
    axes[1].plot(x, [1/r["snapshot"]["market_quotes"]["home"]["ask"] for r in live], ":", color="#8b65ac", label="Synthetic home ask odds (before fees)")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Home win probability (%)")
    for team, key, color in [(payload['home_team'], 'home_decimal_odds', '#2b5b9a'), (payload['away_team'], 'away_decimal_odds', '#b76a38')]:
        axes[1].plot(x, [r["snapshot"][key] for r in live], color=color, linewidth=2, label=team + " fair odds")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Fair decimal odds (log scale)")
    axes[1].yaxis.set_major_formatter(ScalarFormatter())
    for axis in axes:
        for q in range(4):
            axis.axvspan(q * 12, (q + 1) * 12, color="#edf3f7" if q % 2 else "#f8fafb", zorder=-2)
            axis.text(q * 12 + 6, .96, f"Q{q+1}", transform=axis.get_xaxis_transform(), ha="center", fontsize=12)
        for t in (12, 24, 36):
            axis.axvline(t, color="#cdd9e0", linestyle="--", linewidth=1)
        axis.set_xlim(0, 48)
        axis.set_xticks(range(0, 49, 6))
        axis.grid(axis="y", alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="lower left" if axis == axes[0] else "upper left", bbox_to_anchor=None if axis == axes[0] else (.005, .84), fontsize=9)
        for e in events:
            before, after = ((e['p_home_before'] * 100, e['p_home_after'] * 100) if axis == axes[0] else (e['home_odds_before'], e['home_odds_after']))
            xx = e["elapsed_minutes"]
            axis.axvline(xx, color="#a66a18", linestyle=":", alpha=.6)
            axis.plot([xx, xx], [before, after], color="#a66a18", linewidth=2.3)
            axis.scatter([xx, xx], [before, after], facecolors=["white", "#a66a18"], edgecolors="#a66a18", s=30, zorder=4)
            if axis == axes[1]:
                axis.plot([xx, xx], [e['away_odds_before'], e['away_odds_after']], color="#a66a18", linewidth=2.3)
                axis.scatter([xx, xx], [e['away_odds_before'], e['away_odds_after']], facecolors=["white", "#a66a18"], edgecolors="#a66a18", s=30, zorder=4)
            # Offset badges when distinct observations share a stopped game clock.
            dx = .7 * sum(other['number'] < e['number'] and other['elapsed_minutes'] == xx for other in events)
            axis.text(xx + dx, .8 if e['number'] % 2 else .68, str(e['number']), transform=axis.get_xaxis_transform(), ha="center", color="white", fontsize=10,
                      bbox=dict(boxstyle="circle,pad=.3", facecolor="#142b39", edgecolor="none"))
    axes[1].set_xlabel("Elapsed game minutes | quarter clock runs from 12:00 to 00:00")
    table_ax = fig.add_subplot(grid[2]); table_ax.axis("off")
    names = {"因伤离场": "Injury exit", "确认无法回归": "Ruled out", "确认回归／更正": "Return / correction", "对手被驱逐": "Opponent ejected", "次级回归误报被保留": "Conflicting return ignored"}
    cells = [[f"{e['number']}  {e['clock']}", names[e['label']], f"{e['p_home_before']:.1%} > {e['p_home_after']:.1%} ({e['home_delta_pp']:+.2f} pp)",
              f"{e['home_odds_before']:.3f} > {e['home_odds_after']:.3f}", f"{e['away_odds_before']:.3f} > {e['away_odds_after']:.3f}"] for e in events]
    table = table_ax.table(cellText=cells, colLabels=["Game clock", "News event", "Home probability", "Home odds", "Away odds"], colWidths=[.12,.22,.29,.18,.19], cellLoc="left", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dbe3e9"); cell.set_facecolor("#edf3f7" if row == 0 else "white")
    final = payload['steps'][-1]['snapshot']['score']
    fig.supxlabel(f"SYNTHETIC minute samples and incidents; not actual NBA play-by-play. Same-state model effects, not measured causality.\nFinal {payload['home_team']} {final['home_score']} - {final['away_score']} {payload['away_team']}; final settlement is separate from forecasts.", fontsize=10)
    for name in ("sample.png", "overview.png"):
        fig.savefig(output / name, dpi=160)
    fig.savefig(output / "overview.svg")
    plt.close(fig)
