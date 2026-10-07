"""Price discovery against the FIRST official NBA injury report, not just the inactive list.

    python -m evaluation.injury_timing --download      # fetch hourly + 15-min refinement PDFs (cached), then score
    python -m evaluation.injury_timing                 # score from the cache; writes evaluation/results/injury_timing*

The consumer-fairness analysis found that ~91% of the news-direction price move
on 193 test games happened before the inactive list (stamped tip - 30 min). The
inactive list is the last confirmation: the league's official injury report is
published every 15 minutes (Eastern) and lists a game from the afternoon before.
For every one of those games and every player on its inactive list, this finds
the first report that listed the player as Out or Doubtful, then splits the
signed move of the home mid into 24 h anchor -> first report -> inactive list
-> tip.

PDFs are public (https://ak-static.cms.nba.com/referee/injury/) and cached under
the git-ignored data/raw/injury_reports/. A report counts as public REPORT_LAG
after its nominal slot (as data_sources.news does); the nominal time is a
sensitivity. Slots are scanned hourly from noon ET the day before each game
until 11 pm on game day, then the three 15-minute slots before each first
listing are fetched to sharpen it.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data_sources import ET, RAW, name_key, player_lookup

BASE = "https://ak-static.cms.nba.com/referee/injury/"
CACHE = RAW / "injury_reports"
RESULTS = Path(__file__).resolve().parent / "results"
ANCHOR_LEAD = pd.Timedelta(hours=24)
LIST_LEAD = pd.Timedelta(minutes=30)          # inactive list stamp used by consumer_fairness
REPORT_LAG = pd.Timedelta(minutes=15)
LISTED = {"out", "doubtful"}
DEFINITIONS = ("key", "first_fresh", "last")
REPS, SEED = 2000, 7606


# ---------------------------------------------------------------------- report slots and files

def slot_times(dates) -> list:
    """Hourly Eastern slots (naive) from noon the day before each game date to 11 pm on the day."""
    slots = set()
    for d in dates:
        day = pd.Timestamp(d)
        slots |= set(pd.date_range(day - pd.Timedelta(hours=12), day + pd.Timedelta(hours=23), freq="h"))
    return sorted(slots)


def report_name(slot: pd.Timestamp) -> str:
    """2025-26 filename for a naive Eastern slot, e.g. Injury-Report_2026-02-01_05_00PM.pdf."""
    h12, ampm = (slot.hour % 12) or 12, "AM" if slot.hour < 12 else "PM"
    return f"Injury-Report_{slot:%Y-%m-%d}_{h12:02d}_{slot.minute:02d}{ampm}.pdf"


def slot_of(name: str) -> pd.Timestamp:
    """Inverse of report_name (naive Eastern)."""
    day, hh, rest = name.removeprefix("Injury-Report_").removesuffix(".pdf").split("_")
    return pd.Timestamp(f"{day} {hh}:{rest[:2]} {rest[2:]}")


def fetch(name: str, cache: Path = CACHE, timeout=30):
    """Cached path of one report, or None when the league did not publish it (a .missing marker is kept)."""
    import requests

    path, marker = cache / name, cache / f"{name}.missing"
    if path.exists():
        return path
    if marker.exists():
        return None
    try:
        r = requests.get(BASE + name, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (research; nba-agent)"})
    except requests.RequestException:
        return None                                   # transient: retried on the next run, never marked
    if r.status_code == 200 and r.content[:4] == b"%PDF":
        path.write_bytes(r.content)
        return path
    if r.status_code in (403, 404):
        marker.touch()
    return None


def fetch_all(slots, cache: Path = CACHE, workers=8) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    names = sorted({report_name(s) for s in slots})
    with ThreadPoolExecutor(workers) as pool:
        return dict(zip(names, pool.map(lambda n: fetch(n, cache), names)))


def parse_cache(cache: Path = CACHE) -> pd.DataFrame:
    """Every cached PDF parsed to rows plus its slot (memoised in parsed.parquet, keyed by filename)."""
    from data_sources.news import parse_pdf

    store = cache / "parsed.parquet"
    old = pd.read_parquet(store) if store.exists() else pd.DataFrame(columns=["file"])
    done, new = set(old.file), []
    for path in sorted(cache.glob("Injury-Report_*.pdf")):
        if path.name in done:
            continue
        try:
            rows = parse_pdf(path)
        except Exception as exc:                       # a corrupt PDF is reported, never guessed
            print(f"  could not parse {path.name}: {exc.__class__.__name__}")
            rows = []
        new.append(pd.DataFrame(rows).assign(file=path.name) if rows else pd.DataFrame({"file": [path.name]}))
    out = pd.concat([old] + new, ignore_index=True) if new else old
    if new:
        out.to_parquet(store, index=False)
    out = out.dropna(subset=["PlayerName"]).copy()
    out["slot"] = out.file.map(slot_of)
    return out


# ---------------------------------------------------------------------- matching and first listings

def tidy(raw: pd.DataFrame, games: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Parsed report rows matched to games (date + matchup) and players (name key); public_at in UTC."""
    r = raw.copy()
    r["date"] = pd.to_datetime(r.GameDate, format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    r[["away_team", "home_team"]] = r.Matchup.str.split("@", n=1, expand=True)
    lookup = player_lookup(players)
    r["player_id"] = r.PlayerName.map(lambda n: lookup.get(name_key(n)))
    r = r.merge(games[["game_id", "date", "home_team", "away_team"]], on=["date", "home_team", "away_team"])
    r = r.dropna(subset=["player_id"]).astype({"player_id": int})
    r["status"] = r.CurrentStatus.str.lower()
    r["public_at"] = pd.DatetimeIndex(r.slot).tz_localize(ET).tz_convert("UTC") + REPORT_LAG
    return r[["game_id", "player_id", "slot", "public_at", "status", "PlayerName", "Reason", "file"]]


def first_listing(rows: pd.DataFrame) -> pd.DataFrame:
    """First Out/Doubtful slot per (game, player), and when it was public."""
    listed = rows[rows.status.isin(LISTED)].sort_values("slot")
    first = listed.groupby(["game_id", "player_id"], as_index=False).first()
    return first.rename(columns={"slot": "first_slot", "public_at": "first_public", "status": "first_status"})[
        ["game_id", "player_id", "first_slot", "first_public", "first_status"]]


def refine_slots(first: pd.DataFrame) -> list:
    """The three 15-minute slots before each first listing (an hourly scan only bounds it to that hour)."""
    return sorted({t - pd.Timedelta(minutes=m) for t in first.first_slot.dropna() for m in (15, 30, 45)})


def report_time(players: pd.DataFrame, tip, how="key"):
    """One official-report time per game (UTC), or NaT when no absent player was listed Out/Doubtful.

    key:         first listing of the absent player with the most minutes over his last 10 games
    first_fresh: earliest listing after the 24 h anchor (else the earliest listing)
    last:        latest first listing among the game's absent players
    """
    p = players.dropna(subset=["first_public"])
    if p.empty:
        return pd.NaT
    if how == "key":
        return p.sort_values("usual_min", ascending=False).first_public.iloc[0]
    if how == "first_fresh":
        fresh = p.first_public[p.first_public > pd.Timestamp(tip) - ANCHOR_LEAD]
        return fresh.min() if len(fresh) else p.first_public.min()
    if how == "last":
        return p.first_public.max()
    raise ValueError(how)


# ---------------------------------------------------------------------- price moves

def mid_at(prices: tuple, t) -> float:
    """Last mid at or before t (the first one if t precedes every quote). prices = (ns timestamps, mids)."""
    ts, mid = prices
    k = int(np.searchsorted(ts, pd.Timestamp(t).as_unit("ns").value, "right"))
    return float(mid[max(k, 1) - 1])


def split_move(prices: tuple, start, report, listed, tip, sign: float) -> dict:
    """Signed move (news direction) from start to tip, split at the report and at the inactive list.

    A report before `start` is already in the start price (pre_report = 0); a report after the list, or none,
    is treated as arriving with the list (report_to_list = 0).
    """
    report = listed if pd.isna(report) else min(max(pd.Timestamp(report), pd.Timestamp(start)), pd.Timestamp(listed))
    m0, mr, ml, mt = (mid_at(prices, t) for t in (start, report, listed, tip))
    out = {"pre_report": (mr - m0) * sign, "report_to_list": (ml - mr) * sign, "after_list": (mt - ml) * sign}
    out["total"] = (mt - m0) * sign
    return out


WINDOW_EDGES = (-120, -60, -30, -15, 0, 15, 30, 60, 120)     # minutes around the nominal report slot
WINDOW_LABELS = ("anchor → −2 h", "−2 h → −1 h", "−1 h → −30 min", "−30 → −15 min", "−15 min → slot",
                 "slot → +15 min", "+15 → +30 min", "+30 → +60 min", "+1 h → +2 h", "+2 h → inactive list",
                 "inactive list → tip")


def event_window(prices: tuple, start, slot, listed, tip, sign: float) -> np.ndarray:
    """Signed move in each WINDOW_LABELS interval around a report slot (clamped to [start, listed]); sums to the total."""
    pts = [start] + [min(max(slot + pd.Timedelta(minutes=m), start), listed) for m in WINDOW_EDGES] + [listed, tip]
    return np.diff([mid_at(prices, t) for t in pts]) * sign


def shares(d: pd.DataFrame) -> dict:
    total = d.total.sum()
    return {c: float(d[c].sum() / total) if total > 0 else np.nan for c in ("pre_report", "report_to_list", "after_list")}


def share_ci(d: pd.DataFrame, col: str, reps=REPS, seed=SEED) -> tuple:
    """95% CI for one segment's share of the net move, resampling games."""
    x, tot = d[col].to_numpy(float), d.total.to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(d), (reps, len(d)))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = x[idx].sum(1) / tot[idx].sum(1)
    b = b[np.isfinite(b)]
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


# ---------------------------------------------------------------------- pipeline

def load_inputs():
    from data_sources import read_table
    from replay import load_tables

    tables = load_tables()
    players = read_table("players")
    games = pd.read_csv(RESULTS / "fairness_price_discovery_games.csv", dtype={"game_id": str})
    g = tables["games"].assign(game_id=lambda x: x.game_id.astype(str))
    games = games.merge(g[["game_id", "tip_time", "home_team", "away_team"]], on="game_id")
    news = tables["news"].assign(game_id=lambda x: x.game_id.astype(str))
    news = news[news.game_id.isin(games.game_id) & news.status.str.lower().isin(LISTED)]
    absent = news[["game_id", "player_id"]].drop_duplicates().merge(games[["game_id", "tip_time"]], on="game_id")
    pg = tables["player_games"].assign(game_id=lambda x: x.game_id.astype(str)).merge(
        g[["game_id", "tip_time"]], on="game_id").sort_values("tip_time")
    by_player = {k: v for k, v in pg.groupby("player_id")}
    usual = []
    for r in absent.itertuples():
        past = by_player.get(r.player_id, pg.iloc[0:0])
        past = past[past.tip_time < r.tip_time].tail(10)
        usual.append(float(past["min"].mean()) if len(past) else 0.0)
    absent["usual_min"] = usual
    absent["player_name"] = absent.player_id.map(dict(zip(players.player_id, players.player_name)))
    prices = {}
    for t, p in tables["prices"][tables["prices"].market_ticker.isin(games.ticker)].groupby("market_ticker"):
        p = p.sort_values("ts")
        prices[t] = (pd.DatetimeIndex(p.ts).tz_convert("UTC").as_unit("ns").asi8, ((p.bid + p.ask) / 2).to_numpy(float))
    return games, absent, players, prices


def download(games: pd.DataFrame, players: pd.DataFrame):
    hourly = fetch_all(slot_times(games.date.unique()))
    print(f"hourly: {sum(v is not None for v in hourly.values())} of {len(hourly)} report slots published")
    first = first_listing(tidy(parse_cache(), games, players))
    extra = fetch_all([s for s in refine_slots(first) if report_name(s) not in hourly])
    print(f"15-min refinement: {sum(v is not None for v in extra.values())} of {len(extra)} slots published")


def build(games, absent, players, prices, lag=REPORT_LAG) -> tuple:
    rows = tidy(parse_cache(), games, players)
    first = first_listing(rows)
    first["first_public"] = first.first_public + (lag - REPORT_LAG)
    seen = rows.groupby(["game_id", "player_id"]).public_at.min().rename("first_seen").reset_index()
    listing = absent.merge(first, on=["game_id", "player_id"], how="left").merge(seen, on=["game_id", "player_id"],
                                                                                  how="left")
    out = []
    for g in games.itertuples():
        tip = pd.Timestamp(g.tip_time)
        start, listed = tip - ANCHOR_LEAD, tip - LIST_LEAD
        pl = listing[listing.game_id == g.game_id]
        row = {"game_id": g.game_id, "date": g.date, "matchup": f"{g.away_team}@{g.home_team}", "tip_time": tip,
               "news_shift": g.news_shift, "absent_players": len(pl), "seen_in_reports": int(pl.first_seen.notna().sum()),
               "listed_out_doubtful": int(pl.first_public.notna().sum())}
        if g.ticker not in prices:
            out.append(row)
            continue
        for how in DEFINITIONS:
            t = report_time(pl, tip, how)
            s = split_move(prices[g.ticker], start, t, listed, tip, float(np.sign(g.news_shift)))
            row.update({f"{k}_{how}": v for k, v in s.items()})
            row[f"report_{how}"] = t
            row[f"lead_h_{how}"] = (listed - t).total_seconds() / 3600 if pd.notna(t) else np.nan
            row[f"category_{how}"] = ("no listing" if pd.isna(t) else "before anchor" if t <= start
                                      else "after list" if t >= listed else "between")
        out.append(row)
    return pd.DataFrame(out), listing


def summarise(d: pd.DataFrame, how: str) -> dict:
    cols = {f"{k}_{how}": k for k in ("pre_report", "report_to_list", "after_list", "total")}
    x = d.dropna(subset=list(cols)).rename(columns=cols)
    cat = x[f"category_{how}"].value_counts()
    out = {"definition": how, "games": len(x), **{f"games_{k.replace(' ', '_')}": int(cat.get(k, 0))
                                                  for k in ("before anchor", "between", "after list", "no listing")},
           "median_lead_h": float(x[f"lead_h_{how}"].median()),
           "mean_total_pts": float(x.total.mean() * 100),
           **{f"mean_{k}_pts": float(x[k].mean() * 100) for k in ("pre_report", "report_to_list", "after_list")},
           "share_before_list": float((x.pre_report + x.report_to_list).sum() / x.total.sum())}
    for k, v in shares(x).items():
        lo, hi = share_ci(x, k)
        out.update({f"share_{k}": v, f"share_{k}_lo": lo, f"share_{k}_hi": hi})
    fresh = x[x[f"category_{how}"] == "between"]
    out["fresh_games"] = len(fresh)
    out["fresh_median_lead_h"] = float(fresh[f"lead_h_{how}"].median()) if len(fresh) else np.nan
    out.update({f"fresh_share_{k}": v for k, v in (shares(fresh) if len(fresh) else {}).items()})
    return out


def windows(d: pd.DataFrame, games: pd.DataFrame, prices: dict, how="key") -> pd.DataFrame:
    """Share of the net move per window around the nominal slot, on games whose report fell between anchor and list.

    `d` must come from build(..., lag=0) so report_<how> is the nominal slot time.
    """
    ticker = games.set_index("game_id").ticker
    x = d[d[f"category_{how}"] == "between"]
    moves = np.array([event_window(prices[ticker[r.game_id]], r.tip_time - ANCHOR_LEAD, getattr(r, f"report_{how}"),
                                   r.tip_time - LIST_LEAD, r.tip_time, float(np.sign(r.news_shift)))
                      for r in x.itertuples()])
    if not len(moves):
        return pd.DataFrame(columns=["window", "share", "mean_pts"])
    return pd.DataFrame({"window": WINDOW_LABELS, "share": moves.sum(0) / moves.sum(),
                         "mean_pts": moves.mean(0) * 100, "games": len(moves)})


def figure(d: pd.DataFrame, summary: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    a = ax[0]
    labels = {"key": "Key absent player\n(most minutes)", "first_fresh": "Earliest listing\nafter the anchor",
              "last": "Last absent player\nto be listed"}
    segs = [("pre_report", "24 h anchor → first official report", "#2563eb"),
            ("report_to_list", "official report → inactive list (tip − 30 min)", "#16a34a"),
            ("after_list", "inactive list → tip", "#ea580c")]
    s = summary.set_index("definition")
    y = np.arange(len(s))[::-1]
    left = np.zeros(len(s))
    for k, lab, c in segs:
        v = s[f"share_{k}"].to_numpy() * 100
        a.barh(y, v, left=left, color=c, label=lab)
        for yi, l, w in zip(y, left, v):
            if abs(w) >= 5:
                a.text(l + w / 2, yi, f"{w:.0f}%", ha="center", va="center", color="white", fontsize=10, weight="bold")
        left += v
    a.set_yticks(y, [f"{labels[i]}\n({int(s.loc[i, 'games'])} games)" for i in s.index], fontsize=9)
    a.axvline(0, color="black", lw=0.8)
    before = s.loc["key", "share_pre_report"] if "key" in s.index else s["share_pre_report"].iloc[0]
    a.set(xlabel="share of the net news-direction price move, 24 h before tip → tip (%)",
          title=f"{before:.0%} of the move comes BEFORE the first official injury report\n"
                "(key player; report public 15 min after its slot)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9)

    a = ax[1]
    lead = d.lead_h_key.dropna()
    a.hist(lead.clip(upper=48), bins=np.arange(0, 49, 2), color="#475569")
    a.axvline(23.5, color="#2563eb", ls="--", lw=1, label="24 h anchor (tip − 24 h)")
    a.set(xlabel="hours the official report preceded the inactive list (key player; capped at 48)", ylabel="games",
          title=f"Official report lead over the inactive list: median {lead.median():.1f} h")
    a.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def markdown(summary: pd.DataFrame, nominal: pd.DataFrame, cov: dict, win: pd.DataFrame) -> str:
    def pct(v):
        return "–" if pd.isna(v) else f"{v:.0%}"

    lines = ["# Price discovery against the first official NBA injury report", "",
             "Generated by `python -m evaluation.injury_timing`. Games: the 193 test games in "
             "`fairness_price_discovery_games.csv` (at least one player ruled out, M4 moved ≥ 1 pt). The home mid's "
             "move from 24 h before tip to tip is signed in the direction of M4's news shift and split at the first "
             "official injury report and at the inactive list (tip − 30 min). A report counts as public 15 min after "
             "its nominal slot; one published before the 24 h anchor is already in the anchor price "
             "(anchor → report = 0). Shares are of the summed net move; CIs resample games (2,000 replicates).", "",
             "## Coverage", "",
             f"- report PDFs cached: {cov['pdfs']}; hourly slots requested {cov['hourly_requested']}, published "
             f"{cov['hourly_published']}",
             f"- absent players (inactive list) on these games: {cov['players']}; found in some report for that game: "
             f"{cov['seen']} ({cov['seen'] / cov['players']:.0%}); listed Out/Doubtful: {cov['listed']} "
             f"({cov['listed'] / cov['players']:.0%})",
             f"- games with at least one absent player listed Out/Doubtful: {cov['games_listed']} of {cov['games']}", "",
             "## Where the move happens", "",
             "| Report time | Games | Report before anchor / between / after list / none | Median lead over list (h) "
             "| Anchor → report | Report → list | List → tip |", "|" + " --- |" * 7]
    for r in summary.itertuples():
        cells = [f"{pct(getattr(r, f'share_{c}'))} [{pct(getattr(r, f'share_{c}_lo'))}, {pct(getattr(r, f'share_{c}_hi'))}]"
                 for c in ("pre_report", "report_to_list", "after_list")]
        lines.append(f"| {r.definition} | {r.games} | {r.games_before_anchor} / {r.games_between} / "
                     f"{r.games_after_list} / {r.games_no_listing} | {r.median_lead_h:.1f} | " + " | ".join(cells) + " |")
    lines += ["", "Mean signed move per game, pts (anchor → report / report → list / list → tip; total): " + "; ".join(
        f"{r.definition}: {r.mean_pre_report_pts:+.2f} / {r.mean_report_to_list_pts:+.2f} / {r.mean_after_list_pts:+.2f}; "
        f"{r.mean_total_pts:+.2f}" for r in summary.itertuples()),
        f"Share before the inactive list (the consumer-fairness number, recomputed here): "
        f"{summary.share_before_list.iloc[0]:.0%}.", "",
        "### Only games whose first report came between the anchor and the list (fresh official news)", "",
        "| Report time | Games | Median lead (h) | Anchor → report | Report → list | List → tip |",
        "| --- | --- | --- | --- | --- | --- |"]
    for r in summary.itertuples():
        lines.append(f"| {r.definition} | {r.fresh_games} | {r.fresh_median_lead_h:.1f} | {pct(r.fresh_share_pre_report)} "
                     f"| {pct(r.fresh_share_report_to_list)} | {pct(r.fresh_share_after_list)} |")
    lines += ["", "### Sensitivity: report public at its nominal slot (no 15-minute lag)", "",
              "| Report time | Anchor → report | Report → list | List → tip |", "| --- | --- | --- | --- |"]
    for r in nominal.itertuples():
        lines.append(f"| {r.definition} | {pct(r.share_pre_report)} | {pct(r.share_report_to_list)} | "
                     f"{pct(r.share_after_list)} |")
    if len(win):
        lines += ["", f"### Event window around the key player's first report (nominal slot; {int(win.games.iloc[0])} "
                  "games whose report fell between the anchor and the list)", "",
                  "The slot is the first 15-minute report listing the player Out/Doubtful, so his status changed "
                  "between the previous slot and this one.", "", "| Window | Share of net move | Mean move (pts) |",
                  "| --- | --- | --- |"]
        lines += [f"| {r.window} | {r.share:.0%} | {r.mean_pts:+.2f} |" for r in win.itertuples()]
    lines += ["", "**key**: first Out/Doubtful listing of the absent player with the most minutes over his last 10 "
              "games. **first_fresh**: earliest listing after the 24 h anchor (else the earliest). **last**: the "
              "latest first listing among the game's absent players. Only official league reports are used; team "
              "beat reporters and social media can be earlier still, so this bounds how early public news was.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="fetch report PDFs into data/raw/injury_reports first")
    args = ap.parse_args()
    games, absent, players, prices = load_inputs()
    if args.download:
        download(games, players)
    d, listing = build(games, absent, players, prices)
    summary = pd.DataFrame([summarise(d, h) for h in DEFINITIONS])
    dn, _ = build(games, absent, players, prices, lag=pd.Timedelta(0))
    nominal = pd.DataFrame([summarise(dn, h) for h in DEFINITIONS])
    win = windows(dn, games, prices)
    hourly = {report_name(s) for s in slot_times(games.date.unique())}
    pdfs = {p.name for p in CACHE.glob("Injury-Report_*.pdf")}
    cov = {"pdfs": len(pdfs), "hourly_requested": len(hourly), "hourly_published": len(hourly & pdfs),
           "players": len(listing), "seen": int(listing.first_seen.notna().sum()),
           "listed": int(listing.first_public.notna().sum()), "games": len(games),
           "games_listed": int(listing.dropna(subset=["first_public"]).game_id.nunique())}
    RESULTS.mkdir(parents=True, exist_ok=True)
    d.to_csv(RESULTS / "injury_timing_games.csv", index=False)
    listing.to_csv(RESULTS / "injury_timing_players.csv", index=False)
    pd.concat([summary.assign(report_lag_min=15), nominal.assign(report_lag_min=0)]).to_csv(
        RESULTS / "injury_timing.csv", index=False)
    win.to_csv(RESULTS / "injury_timing_window.csv", index=False)
    figure(d, summary, RESULTS / "injury_timing.png")
    (RESULTS / "injury_timing.md").write_text(markdown(summary, nominal, cov, win))
    print((RESULTS / "injury_timing.md").read_text())


if __name__ == "__main__":
    main()
