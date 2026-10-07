"""I: episodic retrieval memory for the market agent.

Every settled decision becomes an episode: the situation the trader saw (price paid,
edge after fees, market move since the anchor, size of the news shift, hours to tip,
news age), the decision, its closing-line value and a one-line lesson. At decision time
MemoryAgent recalls the k most similar episodes that had settled before `now` and
vetoes a candidate trade when those precedents clearly lost value:

    at least MIN_EPISODES settled episodes in memory, k neighbours, and
    neighbours' mean CLV <= SKIP_CLV  ->  no trade ("memory: ...")

Memory can only remove a trade, like the analyst. A vetoed candidate is still stored,
as a "shadow" episode, once its game is final (its CLV is the closing mid minus the
price it would have paid), so a veto can be un-learned instead of becoming permanent.

No leakage: an episode carries settled_at = the game's final_at, and recall only sees
episodes with settled_at <= now. Similarity is Euclidean distance on features
z-scored with the mean and spread of the episodes visible at `now`.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from agents.graph import MarketAgent, log
from replay import AsOf

FEATURES = ("side_price", "gap", "market_move", "model_shift", "hours_to_tip", "news_age")
K = 20
MIN_EPISODES = 40
SKIP_CLV = -0.01


def features(situation: dict) -> np.ndarray:
    age = situation.get("news_age_minutes", np.nan)
    row = {**situation, "news_age": math.log1p(min(float(age), 600.0)) if age == age else np.nan}
    return np.array([float(row.get(f, np.nan)) for f in FEATURES])


def _ts(value) -> pd.Timestamp:
    """UTC-aware timestamp so as-of comparisons never mix naive and aware values."""
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def lesson(situation: dict, side: str, price: float, close: float, clv: float) -> str:
    verdict = "the price moved against it; avoid" if clv < 0 else "the price held or moved our way"
    return (f"bought {side} on {situation.get('team', '?')} at {price:.2f} with edge {situation.get('gap', 0):+.3f}, "
            f"market moved {situation.get('market_move', 0):+.3f}, news {situation.get('news_age_minutes', 0):.0f} "
            f"min old; closed {close:.2f} (CLV {clv:+.3f}): {verdict}")


class EpisodicMemory:
    def __init__(self, episodes=None):
        self.episodes = list(episodes or [])

    def add(self, situation: dict, *, settled_at, as_of, market_ticker, side, price, contracts, clv,
            close_price, shadow=False):
        if settled_at is None or pd.isna(settled_at) or pd.isna(clv):
            return None
        settled_at, as_of = _ts(settled_at), _ts(as_of)
        if settled_at < as_of:
            raise ValueError("an episode cannot settle before its decision")
        ep = {"episode_id": f"e{len(self.episodes) + 1:05d}", "as_of": as_of,
              "settled_at": settled_at, "market_ticker": market_ticker, "side": side,
              "price": float(price), "contracts": float(contracts), "clv": float(clv),
              "close_price": float(close_price), "shadow": bool(shadow),
              "situation": {k: situation.get(k) for k in ("team", "opponent", "market_kind", "hours_to_tip",
                                                           "news_age_minutes", "side_price", "gap", "market_move",
                                                           "model_shift")},
              "lesson": lesson(situation, side, price, close_price, clv)}
        self.episodes.append(ep)
        return ep

    def visible(self, now) -> list:
        now = _ts(now)
        return [e for e in self.episodes if e["settled_at"] <= now]

    def recall(self, situation: dict, now, k: int = K) -> list:
        """The k settled episodes nearest to `situation`, using only episodes settled at or before `now`."""
        seen = self.visible(now)
        if not seen:
            return []
        x = np.vstack([features(e["situation"]) for e in seen])
        mu, sd = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        z = np.nan_to_num((x - mu) / sd)
        q = np.nan_to_num((features(situation) - mu) / sd)
        dist = np.sqrt(((z - q) ** 2).sum(axis=1))
        order = sorted(range(len(seen)), key=lambda i: (dist[i], seen[i]["settled_at"], seen[i]["episode_id"]))
        return [{**seen[i], "distance": float(dist[i])} for i in order[:k]]

    @staticmethod
    def summarise(neighbours: list) -> dict:
        clv = np.array([e["clv"] for e in neighbours], float)
        if len(clv) == 0:
            return {"n": 0, "mean_clv": float("nan"), "se": float("nan")}
        se = float(clv.std(ddof=1) / math.sqrt(len(clv))) if len(clv) > 1 else float("nan")
        return {"n": len(clv), "mean_clv": float(clv.mean()), "se": se,
                "share_negative": float((clv < 0).mean())}

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"episodes": self.episodes}, default=str, indent=1))
        return path

    @classmethod
    def load(cls, path: Path) -> "EpisodicMemory":
        eps = json.loads(Path(path).read_text())["episodes"]
        for e in eps:
            e["as_of"], e["settled_at"] = pd.Timestamp(e["as_of"]), pd.Timestamp(e["settled_at"])
        return cls(eps)


class MemoryAgent(MarketAgent):
    """MarketAgent plus an as-of episodic memory that can veto a candidate trade.

    Plugs in by overriding the analyse node (the veto runs after the notebook rules and
    the analyst) and on_day_end (store the day's settled episodes, then the usual review
    if learning is on). Every veto is kept in self.vetoes for evaluation.
    """

    def __init__(self, *args, memory: EpisodicMemory | None = None, k: int = K, min_episodes: int = MIN_EPISODES,
                 skip_clv: float = SKIP_CLV, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = memory if memory is not None else EpisodicMemory()
        self.k, self.min_episodes, self.skip_clv = k, min_episodes, skip_clv
        self.vetoes = []
        self.recalls = 0

    def analyse(self, state):
        out = super().analyse(state)
        now, game = state["now"], state["game"]
        notes = []
        for r in out["candidates"]:
            if not r["act"]:
                continue
            situation = self.situations[(r["ticker"], now)]
            if len(self.memory.visible(now)) < self.min_episodes:
                notes.append(f"{r['team']}: memory has under {self.min_episodes} settled episodes")
                continue
            near = self.memory.recall(situation, now, self.k)
            s = self.memory.summarise(near)
            self.recalls += 1
            r["memory"] = {**s, "episodes": [e["episode_id"] for e in near]}
            if s["mean_clv"] <= self.skip_clv:
                r.update(act=False, why_not=f"memory: {s['n']} similar past trades averaged {s['mean_clv']:+.3f} CLV")
                self.vetoes.append({"as_of": now, "game_id": game.game_id, "ticker": r["ticker"], "side": r["side"],
                                    "price": r["ask"] if r["side"] == "yes" else 1 - r["bid"], "stake": self.stake * r["scale"],
                                    "tip_time": game.tip_time, "final_at": game.final_at, "situation": situation,
                                    "neighbour_mean_clv": s["mean_clv"], "neighbours": s["n"],
                                    "precedent": near[0]["lesson"] if near else ""})
            notes.append(f"{r['team']}: {s['n']} precedents, mean CLV {s['mean_clv']:+.3f}"
                         + (" -> veto" if not r["act"] else ""))
        if notes:
            out["trace"] = out["trace"] + log("memory", "; ".join(notes))
        return out

    def on_day_end(self, day, fills, rp):
        games = rp.t["games"].set_index("game_id")
        if fills is not None and len(fills):
            for f in fills.itertuples():
                situation = self.situations.get((f.market_ticker, f.as_of))
                if situation is None:
                    continue
                self.memory.add(situation, settled_at=games.loc[f.game_id, "final_at"], as_of=f.as_of,
                                market_ticker=f.market_ticker, side=f.side, price=f.price, contracts=f.contracts,
                                clv=f.clv, close_price=f.close_price)
        today = set(games.index[games.date == day])
        for v in [v for v in self.vetoes if v["game_id"] in today and "clv" not in v]:
            q = AsOf(rp.t, v["tip_time"], rp.price_index).quote(v["ticker"])
            if q is None:
                v["clv"] = float("nan")
                continue
            mid = (q.bid + q.ask) / 2
            close = mid if v["side"] == "yes" else 1 - mid
            v["clv"], v["close_price"] = float(close - v["price"]), float(close)
            self.memory.add(v["situation"], settled_at=v["final_at"], as_of=v["as_of"], market_ticker=v["ticker"],
                            side=v["side"], price=v["price"], contracts=math.floor(v["stake"] / v["price"]),
                            clv=v["clv"], close_price=close, shadow=True)
        super().on_day_end(day, fills, rp)
