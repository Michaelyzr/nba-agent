"""Synthetic policy-check slate shared by the demo and regression tests."""
import pandas as pd

TIP1 = pd.Timestamp("2026-02-01T00:30:00Z")
TIP2 = pd.Timestamp("2026-02-02T00:30:00Z")
HOME, AWAY = "SYN-g2-HOM", "SYN-g2-AWY"
NEWS_AT = TIP2 - pd.Timedelta(hours=2)


def make_tables(with_news=True, home_quote=(0.60, 0.62)):
    games = pd.DataFrame({
        "game_id": ["g1", "g2"], "date": ["2026-01-31", "2026-02-01"], "tip_time": [TIP1, TIP2],
        "final_at": [TIP1 + pd.Timedelta(hours=3), TIP2 + pd.Timedelta(hours=3)],
        "home_team_id": [1, 1], "away_team_id": [2, 2], "home_team": ["HOM", "HOM"],
        "away_team": ["AWY", "AWY"], "home_pts": [110, 101], "away_pts": [100, 99]})
    player_games = pd.DataFrame({"game_id": ["g1"], "player_id": [7], "team_id": [1], "min": [34.0], "pts": [25],
                                 "fga": [18], "fta": [6], "usage": [0.3], "started": [True]})
    news = pd.DataFrame({"news_id": ["n_out"], "published_at": [NEWS_AT], "game_id": ["g2"], "player_id": [7],
                         "status": ["Out"], "source": ["test"], "url": [""], "text": ["Player 7 listed out"]})
    if not with_news:
        news = news.iloc[0:0]
    markets = pd.DataFrame({"venue": "synthetic", "market_ticker": [HOME, AWAY], "kind": "game", "game_id": "g2",
                            "team": ["HOM", "AWY"], "player_id": None, "line": None, "title": ["HOM", "AWY"]})
    ts = pd.date_range(TIP2 - pd.Timedelta(hours=3), TIP2 + pd.Timedelta(hours=1), freq="1min")
    bid, ask = home_quote
    prices = pd.concat([
        pd.DataFrame({"venue": "synthetic", "market_ticker": HOME, "ts": ts, "bid": bid, "ask": ask, "volume": 500.0}),
        pd.DataFrame({"venue": "synthetic", "market_ticker": AWAY, "ts": ts, "bid": round(1 - ask, 2),
                      "ask": round(1 - bid, 2), "volume": 500.0})])
    settlements = pd.DataFrame({"market_ticker": [HOME, AWAY], "settled_at": TIP2 + pd.Timedelta(hours=3),
                                "outcome": [1, 0]})
    return {"games": games, "player_games": player_games, "news": news, "markets": markets,
            "prices": prices, "settlements": settlements}
