"""Run both real agent loops with synthetic news/scores; export an offline demo.

    python -m agents.loop_demo --output docs/samples/news_loop

The two agents have separate directories and baselines. No HTTP or model downloads.
"""
import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from agents.graph import HOME_EDGE, clip, log5_home, record_forecaster, win_rate
from agents.inplay import InPlayAgent
from agents.pregame import PregameAgent
from data_sources import ROOT, read_table
from data_sources.inplay import TableInPlay
from data_sources.inplay_demo import synthetic_full_game
from agents.game_timeline import elapsed_seconds, clock_label, timeline_events
from agents.forecast_audit import MODEL_OPTIONS, SOURCES
from agents.market_analysis import synthetic_quotes, compare_routes
from data_sources.inplay_types import event_id, utc
from data_sources.live_news import TableNews, post_rows
from data_sources.inplay_news import text_events
from data_sources.news_registry import NewsRegistry
from forecast.history import History
from replay import AsOf

SAMPLE_OUTPUT = ROOT / "docs" / "samples" / "news_loop"


def records(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


class DemoPregameFeed(TableNews):
    def fetch(self, *args):
        batch = super().fetch(*args)
        batch.items = [{**r, "item_id": r["news_id"], "review_required": r["status"] == "review"} for r in batch.rows]
        batch.coverage = [{"source": "synthetic_demo", "state": "synthetic", "transport": "offline_table"}]
        return batch


class DemoInPlayFeed(TableInPlay):
    def fetch(self, *args):
        batch = super().fetch(*args)
        batch.evidence = [{**r, "item_id": r["event_id"], "review_required": r["status"] == "review"} for r in batch.events]
        batch.coverage = [{"source": "synthetic_demo", "state": "synthetic", "transport": "offline_table"}]
        return batch


class IndependentHistoricalPrior:
    name = "demo-history-record"

    def win(self, view, game):
        done = view.games().dropna(subset=["home_pts"])
        return {"p_home": clip(log5_home(win_rate(done, game.home_team_id), win_rate(done, game.away_team_id)))}


def inputs(tables, players, game):
    tip = utc(game.tip_time)
    view = AsOf(tables, tip, {})
    history = History(view.player_games(), view.games())
    home = history.rotation(game.home_team_id, tip)
    hp = max(home, key=lambda p: home[p][0])
    roster = set(history.roster(game.home_team_id, tip)) | set(history.roster(game.away_team_id, tip))
    names = dict(zip(players.player_id, players.player_name))
    label = str(names[hp])
    registry = NewsRegistry.load()
    reporter = registry.data["teams"][game.home_team]["reporter"]["handle"].lower()
    definitions = [(-90, f"{label} is questionable tonight with an ankle injury", "espn_rss", "合成权威媒体情景（ESPN层级）"),
                   (-85, f"{label} is available tonight", "x:" + reporter, "合成跟队记者情景"),
                   (-30, f"{label} is available tonight", "espn_rss", "合成权威确认情景（ESPN层级）")]
    rows = []
    for minutes, text, source, source_label in definitions:
        published = tip + pd.Timedelta(minutes=minutes)
        observed = published + pd.Timedelta(seconds=2)
        item = event_id("demo", game.game_id, minutes)
        extracted = post_rows(text, game, players, roster, source, published, observed, "", item)
        rows.extend({**r, "synthetic": True, "source_label": source_label} for r in extracted)
    pre_times = [tip - pd.Timedelta(hours=2)] + [utc(r["observed_at"]) for r in rows] + [tip - pd.Timedelta(minutes=1)]
    pre_labels = ["历史基准", "赛前出场存疑", "次级消息冲突", "权威确认可出场", "重复消息去重"]
    scores, original = synthetic_full_game(tables, players, game)
    # Reuse scenario inputs, then run the actual text parser before the actual agent.
    in_rows = []
    sources = {"left_injured": ("x:shamscharania", "合成 Shams 情景"),
               "injury_out": ("espn_rss", "合成权威媒体情景（ESPN层级）"),
               "returned": ("espn_rss", "合成权威回归确认（ESPN层级）"),
               "ejected": ("nba_live", "合成 NBA 官方比赛事件")}
    for raw in original.to_dict("records"):
        source, source_label = ("x:shamscharania", "合成次级回归误报") if raw["demo_conflict"] else sources[raw["status"]]
        extracted = text_events(raw["text"], game, players, roster, source, raw["published_at"], raw["observed_at"], "", raw["event_id"])
        in_rows.extend({**r, "synthetic": True, "source_label": source_label} for r in extracted)
    in_labels = list(scores.demo_label)
    return pd.DataFrame(rows), pre_times, pre_labels, scores, pd.DataFrame(in_rows), in_labels


def generate_demo(output=SAMPLE_OUTPUT, game_id="401811041"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    folder = ROOT / "data" / "sample"
    tables = {name: read_table(name, folder) for name in ("games", "player_games")}
    tables["news"] = pd.DataFrame()
    players = read_table("players", folder)
    matches = tables["games"][tables["games"].game_id.astype(str) == str(game_id)]
    if len(matches) != 1:
        raise ValueError("demo game must exist uniquely in the committed sample")
    game = next(matches.itertuples(index=False))
    news, pre_times, pre_labels, scores, events, in_labels = inputs(tables, players, game)
    steps = []
    # Fresh isolated scratch state makes regeneration repeatable without deleting runs.
    with tempfile.TemporaryDirectory(prefix="nba-loop-demo-") as scratch:
        pre = PregameAgent(tables, players, game, record_forecaster, DemoPregameFeed(news), Path(scratch) / "pregame")
        for i, (at, label) in enumerate(zip(pre_times, pre_labels)):
            s = pre.poll(at)["snapshot"]
            steps.append({"phase": "pregame", "step": i + 1, "label": label, "snapshot": s})
        assert pre.poll(utc(game.tip_time))["stopped"]
        live = InPlayAgent(tables, players, game, DemoInPlayFeed(scores, events), Path(scratch) / "inplay",
                           prior_model=IndependentHistoricalPrior())
        previous = None
        for i, (at, label) in enumerate(zip(scores.observed_at, in_labels)):
            result = live.poll(at)
            snapshot = result["snapshot"]
            snapshot["market_quotes"] = synthetic_quotes(snapshot, game, previous)
            snapshot["market_comparison"] = compare_routes(snapshot, snapshot["market_quotes"])
            steps.append({"phase": "inplay", "step": i + 1, "label": label, "snapshot": result["snapshot"]})
            previous = snapshot
        assert live.stopped
        for phase in ("pregame", "inplay"):
            # Export actual logs for auditing; do not mix or overwrite runtime directories.
            target = output / "logs" / phase
            target.mkdir(parents=True, exist_ok=True)
            for path in (Path(scratch) / phase).iterdir():
                (target / path.name).write_bytes(path.read_bytes())
    payload = {"schema_version": 3, "synthetic": True, "title": f"{game.away_team} @ {game.home_team} · 新闻 Agent 循环",
               "game_id": str(game.game_id), "home_team": game.home_team, "away_team": game.away_team,
               "tip_time": utc(game.tip_time).isoformat(), "steps": steps,
               "notice": "新闻、比分和记者/媒体情景为合成数据；历史统计仅提供基准与轮换，不代表实际报道或历史实况。两套循环独立计算基准。"}
    payload["timeline_events"] = timeline_events([r for r in steps if r["phase"] == "inplay"])
    payload["model_options"] = MODEL_OPTIONS
    payload["reference_sources"] = SOURCES
    view = AsOf(tables, utc(game.tip_time), {})
    done = view.games().dropna(subset=["home_pts"])
    relevant = done[(done.home_team_id.isin([game.home_team_id, game.away_team_id]))
                    | (done.away_team_id.isin([game.home_team_id, game.away_team_id]))]
    pids = set(scores.iloc[0]["roster"])
    player_rows = view.player_games()
    selected = player_rows[(player_rows.player_id.isin(pids)) & (player_rows["min"] > 0)].copy()
    selected["_tip"] = selected.game_id.map(dict(zip(done.game_id, done.tip_time)))
    selected["historical_final_at"] = selected.game_id.map(dict(zip(done.game_id, done.final_at)))
    selected = selected.dropna(subset=["_tip"]).sort_values("_tip").groupby("player_id").tail(10).rename(columns={"_tip": "historical_tip_time"})
    team_records = []
    for team, team_id in ((game.home_team, game.home_team_id), (game.away_team, game.away_team_id)):
        appearances = relevant[(relevant.home_team_id == team_id) | (relevant.away_team_id == team_id)]
        wins = ((appearances.home_team_id == team_id) & (appearances.home_pts > appearances.away_pts)
                | (appearances.away_team_id == team_id) & (appearances.away_pts > appearances.home_pts)).sum()
        team_records.append({"team": team, "games": len(appearances), "wins": int(wins), "record_win_rate": float(win_rate(done, team_id))})
    payload["original_data"] = {"historical_games": records(relevant),
                                "player_games": records(selected), "team_records": team_records,
                                "home_edge": HOME_EDGE,
                                "prior_equation": "队伍胜率 = (历史胜场 + 5) / (历史场数 + 10)；log5 对阵概率加主场修正后截断，得到开赛独立基准。",
                                "selection": "保留两队全部可见已完赛记录，以及受事件影响球员最近 10 场出场数据；不嵌入其他球队和无关球员的大表。",
                                "source_files": ["data/sample/games.parquet", "data/sample/player_games.parquet", "data/sample/players.parquet"],
                                "input_scores": records(scores), "input_events": records(events),
                                "input_pregame_news": records(news),
                                "history_cutoff": utc(game.tip_time).isoformat()}
    encoded = json.dumps(payload, ensure_ascii=False, default=str, allow_nan=False, indent=2)
    (output / "result.json").write_text(encoded)
    rows = []
    for step in steps:
        s, r = step["snapshot"], step["snapshot"]["report"]
        rows.append({"phase": step["phase"], "step": step["step"], "event": step["label"], "as_of": s["as_of"],
                     "p_home": s["p_home"], "home_decimal_odds": s["home_decimal_odds"],
                     "reference_p_home": r["reference_p_home"], "new_event_effect_pp": r["new_event_effect_pp"],
                     "conflicts": r["conflict_count"],
                     "game_clock": clock_label(s["score"]) if s.get("score") else "",
                     "elapsed_minutes": elapsed_seconds(s["score"]) / 60 if s.get("score") else None,
                     "news_home_odds_delta": (r.get("news_effect") or {}).get("home_odds_delta"),
                     "news_away_odds_delta": (r.get("news_effect") or {}).get("away_odds_delta")})
    pd.DataFrame(rows).to_csv(output / "timeline.csv", index=False)
    (output / "report.md").write_text("# Sample news loop\n\n" + payload["notice"] + "\n\n" +
                                      "\n\n---\n\n".join(s["snapshot"]["report"]["markdown"] for s in steps))
    from agents.loop_visuals import export_visuals
    export_visuals(payload, output)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SAMPLE_OUTPUT)
    parser.add_argument("--game-id", default="401811041")
    args = parser.parse_args()
    result = generate_demo(args.output, args.game_id)
    print(f"Generated {len(result['steps'])} actual agent polls with synthetic inputs: {args.output.resolve()}")
    print("Open demo.html, or: streamlit run loop_demo_app.py")


if __name__ == "__main__":
    main()
