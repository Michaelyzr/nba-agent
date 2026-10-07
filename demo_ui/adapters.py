"""Read-only adapters from the research runtime to presentation-friendly view models.

The module deliberately does not persist state, train models, call live APIs, or
reimplement agent policy.  It packages existing outputs for ``demo_app.py`` and
falls back to the repository's deterministic record forecaster when model
artifacts are not present.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

import replay
from agents import coach, coach_agent
from agents.graph import DEFAULT_MIN_EDGE, MarketAgent, NEWS_WINDOW, fee_per_contract, record_forecaster
from agents.notebook import Notebook
from data_sources import FROZEN, ROOT
from data_sources.inplay_demo import synthetic_game
from data_sources.sample import SAMPLE
from forecast.inplay import InPlayWinModel


@dataclass(frozen=True)
class Scenario:
    game_id: str
    label: str
    note: str


@dataclass
class Runtime:
    root: Path
    folder: Path
    tables: dict[str, pd.DataFrame]
    players: pd.DataFrame
    names: dict[Any, str]
    forecaster: Any
    model_name: str
    model_status: str
    replay: replay.Replay


class RecordForecasterAdapter:
    """Expose the built-in record baseline through the M5 ``Forecaster`` API."""

    name = "record-baseline"

    def win(self, view, game, out=(), availability=None) -> dict:
        markets = view.markets(game.game_id)
        probabilities = record_forecaster(view, game, markets, {"out": list(out)})
        home = markets[(markets.kind == "game") & (markets.team == game.home_team)]
        if home.empty:
            raise ValueError(f"no home-team market for game {game.game_id}")
        ticker = home.iloc[0].market_ticker
        return {
            "game_id": game.game_id,
            "as_of": pd.Timestamp(view.now).isoformat(),
            "model": self.name,
            "p_home": float(probabilities[ticker]),
            "missing": len(out),
            "overrides": {"out": list(out)},
        }

    def __call__(self, view, game, markets: pd.DataFrame, overrides: dict) -> dict:
        return record_forecaster(view, game, markets, overrides)


def load_runtime(root: Path = ROOT) -> Runtime:
    """Load local replay data and the best available forecaster without networking."""
    folder = FROZEN if (FROZEN / "prices.parquet").exists() else SAMPLE
    tables = replay.load_tables(folder)
    players_path = folder / "players.parquet"
    players = pd.read_parquet(players_path) if players_path.exists() else pd.DataFrame(
        columns=["player_id", "player_name"]
    )
    names = dict(zip(players.player_id, players.player_name))
    try:
        from forecast.api import Forecaster

        forecaster = Forecaster.load(root / "models")
        model_name = forecaster.name
        model_status = "TRAINED ARTIFACT"
    except (FileNotFoundError, ImportError, AttributeError, ValueError):
        forecaster = RecordForecasterAdapter()
        model_name = forecaster.name
        model_status = "DETERMINISTIC FALLBACK"
    rp = replay.Replay(tables, lambda *_: [])
    return Runtime(root, folder, tables, players, names, forecaster, model_name, model_status, rp)


def _game(runtime: Runtime, game_id: str):
    games = runtime.tables["games"]
    row = games[games.game_id.astype(str) == str(game_id)]
    if row.empty:
        raise KeyError(f"game {game_id} is not in the replay data")
    return next(row.itertuples(index=False))


def decision_time(runtime: Runtime, game) -> pd.Timestamp:
    """Use the latest pre-tip news event, otherwise the standard one-hour lead."""
    news = runtime.tables["news"]
    recent = news[
        (news.game_id.astype(str) == str(game.game_id))
        & (news.published_at >= game.tip_time - NEWS_WINDOW)
        & (news.published_at < game.tip_time)
    ]
    baseline = pd.Timestamp(game.tip_time) - replay.LEAD
    return max([baseline, *recent.published_at.tolist()])


def scenarios(runtime: Runtime) -> list[Scenario]:
    """A small, legible slate with both action and pass decisions."""
    curated = [
        ("401810572", "UTA at IND · late inactive list", "Agent acts after nine status updates"),
        ("401810576", "BOS at DAL · disciplined pass", "News moves the model, but not enough after costs"),
        ("401811041", "ORL at BOS · confidence meets reality", "A strong estimate is tested by the final result"),
        ("401811051", "PHX at OKC · market already aligned", "Eight updates, no threshold-clearing edge"),
        ("401810575", "ATL at MIA · model-market gap", "Late availability news creates an actionable gap"),
    ]
    available = set(runtime.tables["games"].game_id.astype(str))
    out = [Scenario(*row) for row in curated if row[0] in available]
    if out:
        return out
    traded = runtime.tables["games"]
    traded = traded[traded.game_id.isin(runtime.tables["markets"].game_id)].sort_values("tip_time").tail(5)
    return [
        Scenario(str(g.game_id), f"{g.away_team} at {g.home_team} · {str(g.date)[:10]}", "Historical replay")
        for g in traded.itertuples(index=False)
    ]


def live_scenarios(live_context) -> list[Scenario]:
    """Current, quality-gated Polymarket moneylines exposed as demo choices."""
    if not live_context.usable_for_live_analysis:
        return []
    games = live_context.agent_tables.games.sort_values("tip_time")
    return [
        Scenario(
            str(game.game_id),
            f"{game.away_team} at {game.home_team} · Live Polymarket",
            str(game.event_title or "Current NBA moneyline"),
        )
        for game in games.itertuples(index=False)
    ]


def _backed_team(game, contract_team: str, side: str) -> str:
    if side == "yes":
        return contract_team
    return game.away_team if contract_team == game.home_team else game.home_team


def _oriented(value: float, side: str) -> float:
    return float(value if side == "yes" else 1 - value)


def analyse_scenario(runtime: Runtime, game_id: str) -> dict:
    """Run the existing deterministic agent once and package its auditable outputs."""
    game = _game(runtime, game_id)
    now = decision_time(runtime, game)
    view = replay.AsOf(runtime.tables, now, runtime.replay.price_index)
    agent = MarketAgent(Notebook(), forecaster=runtime.forecaster, learn=False, use_llm=False)
    decision = agent.decide(view, game, now)
    candidates = decision.get("candidates") or []
    orders = decision.get("orders") or []
    chosen_ticker = orders[0].market_ticker if orders else None
    candidate = next((c for c in candidates if c["ticker"] == chosen_ticker), None)
    if candidate is None:
        candidate = max(candidates, key=lambda c: float(c.get("gap", -1e9)), default=None)
    if candidate is None:
        raise ValueError("the selected game has no fresh game-winner market at the decision time")

    side = orders[0].side if orders else candidate["side"]
    backed_team = _backed_team(game, candidate["team"], side)
    snapshot = coach.snapshot(runtime.forecaster, view, game, runtime.names)
    p_home_raw = float(snapshot["p_home_after"])
    raw_probability = p_home_raw if backed_team == game.home_team else 1 - p_home_raw
    market_mid_yes = (float(candidate["bid"]) + float(candidate["ask"])) / 2
    market_mid = _oriented(market_mid_yes, side)
    market_bid = float(candidate["bid"]) if side == "yes" else 1 - float(candidate["ask"])
    market_ask = float(candidate["ask"]) if side == "yes" else 1 - float(candidate["bid"])
    agent_probability = _oriented(float(candidate["p"]), side)
    anchor_probability = _oriented(float(candidate["before"]), side)

    grade = {"filled": False, "status": "passed", "why": candidate.get("why_not") or "no order"}
    if orders:
        fill, why = runtime.replay._fill(orders[0], now, game.tip_time)
        if fill is None:
            grade = {"filled": False, "status": "not filled", "why": why}
        else:
            settled = runtime.replay._settle(
                {
                    **fill,
                    "market_ticker": orders[0].market_ticker,
                    "side": orders[0].side,
                    "p_model": orders[0].p_model,
                    "game_id": game.game_id,
                    "as_of": now,
                },
                game.tip_time,
            )
            grade = {"filled": True, "status": "settled", **settled}

    home_market = runtime.tables["markets"]
    home_market = home_market[
        (home_market.game_id.astype(str) == str(game.game_id))
        & (home_market.kind == "game")
        & (home_market.team == game.home_team)
    ]
    close_home = None
    if not home_market.empty:
        close_quote = replay.AsOf(runtime.tables, game.tip_time, runtime.replay.price_index).quote(
            home_market.iloc[0].market_ticker
        )
        if close_quote is not None:
            close_home = float((close_quote.bid + close_quote.ask) / 2)
    home_won = bool(game.home_pts > game.away_pts)

    price_frame = view.prices(candidate["ticker"]).copy()
    price_frame = price_frame[price_frame.ts >= game.tip_time - pd.Timedelta(hours=24)]
    if not price_frame.empty:
        if side == "yes":
            price_frame = price_frame.assign(
                bid_side=price_frame.bid, ask_side=price_frame.ask, mid=(price_frame.bid + price_frame.ask) / 2
            )
        else:
            price_frame = price_frame.assign(
                bid_side=1 - price_frame.ask,
                ask_side=1 - price_frame.bid,
                mid=1 - (price_frame.bid + price_frame.ask) / 2,
            )

    advice = coach_agent.advise(snapshot, backed_team, history=[], llm=False)
    return {
        "game": game,
        "now": now,
        "minutes_to_tip": int(round((game.tip_time - now).total_seconds() / 60)),
        "decision": decision,
        "candidate": candidate,
        "orders": orders,
        "snapshot": snapshot,
        "backed_team": backed_team,
        "contract_team": candidate["team"],
        "side": side,
        "raw_probability": raw_probability,
        "market_mid": market_mid,
        "market_bid": market_bid,
        "market_ask": market_ask,
        "agent_probability": agent_probability,
        "anchor_probability": anchor_probability,
        "edge": float(candidate["gap"]),
        "grade": grade,
        "price_frame": price_frame,
        "news": view.news(game.game_id).sort_values("published_at"),
        "home_won": home_won,
        "close_home": close_home,
        "model_brier": (p_home_raw - float(home_won)) ** 2,
        "market_brier": None if close_home is None else (close_home - float(home_won)) ** 2,
        "coach": advice,
        "lessons": coach.lessons(snapshot),
        "provenance": "HISTORICAL REPLAY",
        "is_live": False,
    }


def analyse_live_scenario(runtime: Runtime, live_context, game_id: str) -> dict:
    """Run the existing MarketAgent against one current Polymarket snapshot.

    Live mode compares the Agent's statistical forecast directly with the
    current quote because one refresh is not a 24-hour price history. No cached
    quote, historical outcome, or synthetic news is inserted.
    """
    if not live_context.usable_for_live_analysis:
        raise ValueError("live Polymarket context is not usable")
    games = live_context.agent_tables.games
    selected = games[games.game_id.astype(str) == str(game_id)]
    if selected.empty:
        raise KeyError(f"live game {game_id} is not in the current Polymarket snapshot")
    game = next(selected.itertuples(index=False))
    view = live_context.view(runtime.tables)
    agent = MarketAgent(
        Notebook(), forecaster=runtime.forecaster, learn=False, use_llm=False, anchor=False
    )
    decision = agent.decide(view, game, live_context.now)
    candidates = decision.get("candidates") or []
    orders = decision.get("orders") or []
    chosen_ticker = orders[0].market_ticker if orders else None
    candidate = next((row for row in candidates if row["ticker"] == chosen_ticker), None)
    if candidate is None:
        candidate = max(candidates, key=lambda row: float(row.get("gap", -1e9)), default=None)
    if candidate is None:
        raise ValueError("the live game has no fresh, model-covered moneyline")

    side = orders[0].side if orders else candidate["side"]
    backed_team = _backed_team(game, candidate["team"], side)
    market_mid_yes = (float(candidate["bid"]) + float(candidate["ask"])) / 2
    market_mid = _oriented(market_mid_yes, side)
    market_bid = float(candidate["bid"]) if side == "yes" else 1 - float(candidate["ask"])
    market_ask = float(candidate["ask"]) if side == "yes" else 1 - float(candidate["bid"])
    agent_probability = _oriented(float(candidate["p"]), side)
    prices = live_context.agent_tables.prices
    quote = prices[prices.market_ticker == candidate["ticker"]].iloc[-1]
    threshold = float(candidate.get("min_edge", DEFAULT_MIN_EDGE))
    long_shot = market_ask <= coach.LONG_SHOT
    break_even = market_ask + fee_per_contract(market_ask)
    checks = {
        "break_even": {
            "ask": market_ask,
            "fee": fee_per_contract(market_ask),
            "breakeven": break_even,
            "estimate": agent_probability,
            "gap": float(candidate["gap"]),
            "flag": float(candidate["gap"]) < threshold,
            "severe": float(candidate["gap"]) < 0,
        },
        "long_shot": {
            "ask": market_ask,
            "flag": long_shot,
            "severe": long_shot and float(candidate["gap"]) < threshold,
        },
        "priced_in": {"move": 0.0, "flag": False, "severe": False},
    }
    quality = live_context.snapshot.quality
    context_items = [
        f"Live Polymarket quote fetched {pd.Timestamp(quality.fetched_at):%H:%M:%S UTC}.",
        f"Current {backed_team} market: {market_bid:.0%} bid and {market_ask:.0%} ask.",
    ]
    liquidity = getattr(quote, "liquidity", None)
    if liquidity is not None and not pd.isna(liquidity):
        context_items.append(f"Reported market liquidity: ${float(liquidity):,.0f}.")
    else:
        context_items.append("No verified live injury feed is attached to this market snapshot.")
    market_row = live_context.agent_tables.markets[
        live_context.agent_tables.markets.market_ticker == candidate["ticker"]
    ].iloc[0]

    return {
        "game": game,
        "now": live_context.now,
        "minutes_to_tip": max(0, int(round((pd.Timestamp(game.tip_time) - live_context.now).total_seconds() / 60))),
        "decision": decision,
        "candidate": candidate,
        "orders": orders,
        "backed_team": backed_team,
        "contract_team": candidate["team"],
        "side": side,
        "agent_probability": agent_probability,
        "market_mid": market_mid,
        "market_bid": market_bid,
        "market_ask": market_ask,
        "edge": float(candidate["gap"]),
        "coach": {
            "steps": [
                {"check": name, "flag": bool(result["flag"]), "severe": bool(result["severe"])}
                for name, result in checks.items()
            ],
            "results": checks,
        },
        "news": pd.DataFrame(columns=runtime.tables["news"].columns),
        "context_items": context_items[:3],
        "close_home": None,
        "provenance": "LIVE POLYMARKET",
        "is_live": True,
        "fetched_at": quality.fetched_at,
        "polymarket_url": market_row.polymarket_url,
    }


def inplay_story(runtime: Runtime, analysis: dict) -> dict:
    """Use the repository's explicitly synthetic score feed and in-play model."""
    if runtime.players.empty:
        return {"available": False, "reason": "Player roster data is unavailable."}
    try:
        scores, events = synthetic_game(runtime.tables, runtime.players, analysis["game"])
    except (ValueError, KeyError, IndexError) as exc:
        return {"available": False, "reason": str(exc)}
    model = InPlayWinModel()
    prior = float(analysis["snapshot"]["p_home_after"])
    checkpoints = []
    for row in scores.itertuples(index=False):
        score = row._asdict()
        probability = model.predict(score, prior)
        if row.phase == "final":
            clock = "Final"
        else:
            minutes, seconds = divmod(int(row.clock_seconds), 60)
            clock = f"Q{int(row.period)} · {minutes}:{seconds:02d}"
        visible_events = events[events.published_at <= row.observed_at].sort_values("published_at")
        checkpoints.append(
            {
                **score,
                "clock_label": clock,
                "p_home": probability,
                "events": visible_events.to_dict("records"),
            }
        )
    return {
        "available": True,
        "model_name": model.name,
        "prior": prior,
        "checkpoints": checkpoints,
        "provenance": "DEMO / SYNTHETIC",
        "disclosure": "Synthetic score and event sequence; historical rosters and the existing in-play model.",
    }


def research_inventory(runtime: Runtime) -> dict:
    """Summarise shipped modules and recorded evaluation evidence without rerunning studies."""
    results = runtime.root / "evaluation" / "results"
    metrics = []
    m4_path = results / "m4_vs_market.csv"
    if m4_path.exists():
        frame = pd.read_csv(m4_path)
        for name in ("M4, absences known", "Market, 1 h before tip"):
            row = frame[frame.predictor == name]
            if not row.empty:
                metrics.append({"label": name, "value": f"{float(row.iloc[0].brier):.3f}", "unit": "Brier"})
    return {
        "metrics": metrics,
        "groups": [
            {
                "status": "CORE",
                "items": [
                    ("MarketAgent", "Deterministic decision graph used in this demo"),
                    ("Replay + market grading", "As-of prices, fill simulation, CLV and settlement"),
                    ("Coach + League", "Education and paper-practice interfaces"),
                    ("Pregame / In-play", "Existing lifecycle capabilities, surfaced without merging their state"),
                ],
            },
            {
                "status": "EXPERIMENTAL",
                "items": [
                    ("ToolAgent", "Research agent; not used in the product path"),
                    ("Memory + rule DSL", "Learning experiments behind a separate evaluation boundary"),
                    ("Learned policy", "Offline selection experiment; not the live decision policy"),
                ],
            },
            {
                "status": "EVALUATION ONLY",
                "items": [
                    ("Calibration", "Recorded" if (results / "m4_calibration.csv").exists() else "No result file"),
                    ("M6 ImpactNet", "Recorded" if (results / "m6_heldout.csv").exists() else "No result file"),
                    ("LLM agent arms", "Partial result set" if (results / "llm_agent.csv").exists() else "No result file"),
                    ("Neural win model", "Recorded" if (results / "win_nn.md").exists() else "No result file"),
                ],
            },
        ],
    }
