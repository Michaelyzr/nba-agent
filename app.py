"""E4: Streamlit demo of one replayed night across channels (README section 7).

    streamlit run app.py

Reads data/frozen/ if it has prices, otherwise the committed data/sample/.
Models come from models/ (python -m forecast.train); runs from runs/.
"""
import copy
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import replay
from agents import coach, league
from agents.briefs import channel_briefs, outlook
from agents.graph import FAULTS, OUT_STATUSES, MarketAgent
from agents.notebook import Notebook
from data_sources import FROZEN, ROOT

SAMPLE = ROOT / "data" / "sample"
RESULTS = ROOT / "evaluation" / "results"
st.set_page_config(page_title="NBA late-news agent", layout="wide")


@st.cache_resource
def load():
    folder = FROZEN if (FROZEN / "prices.parquet").exists() else SAMPLE
    tables = replay.load_tables(folder)
    players = pd.read_parquet(folder / "players.parquet") if (folder / "players.parquet").exists() else None
    names = dict(zip(players.player_id, players.player_name)) if players is not None else {}
    return folder, tables, names


@st.cache_resource
def forecaster():
    from forecast.api import Forecaster
    return Forecaster.load()


def runs() -> list:
    return sorted({p.parent for p in replay.RUNS.rglob("trace.jsonl")}, key=lambda p: p.stat().st_mtime, reverse=True)


def traces(run: Path) -> list:
    return [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()]


def engine(tables) -> replay.Replay:
    rp = st.session_state.setdefault("rp", {})
    if id(tables) not in rp:
        rp[id(tables)] = replay.Replay(tables, lambda *a: [])
    return rp[id(tables)]


def view_at(tables, now):
    return replay.AsOf(tables, now, engine(tables).price_index)


def price_chart(view, ticker, game, news, title):
    p = view.prices(ticker)
    p = p[p.ts >= pd.Timestamp(game.tip_time) - pd.Timedelta(hours=7)].assign(mid=lambda d: (d.bid + d.ask) / 2)
    chart = alt.Chart(p).mark_line().encode(x=alt.X("ts:T", title="time (UTC)", scale=alt.Scale(type="utc")),
                                            y=alt.Y("mid:Q", title=title, scale=alt.Scale(zero=False)))
    marks = [alt.Chart(pd.DataFrame({"ts": news.published_at})).mark_rule(color="orange").encode(x="ts:T")]
    if view.now >= game.tip_time:
        marks.append(alt.Chart(pd.DataFrame({"ts": [game.tip_time]})).mark_rule(color="red").encode(x="ts:T"))
    return alt.layer(chart, *marks)


def out_list(news):
    return news.loc[news.status.astype(str).str.lower().isin(OUT_STATUSES), "player_id"].tolist()


def run_night(tables, day, notebook, game_id=None, plant=None, channel="platform", confirm=None):
    kwargs = {"confirm": confirm} if confirm else {}
    agent = MarketAgent(notebook, forecaster=forecaster(), learn=False, plant=plant, channel=channel, **kwargs)
    policy = agent.policy if game_id is None else (lambda v, g, n: agent.policy(v, g, n) if g.game_id == game_id else [])
    decisions, fills = replay.Replay(tables, policy).run(day, day)
    return agent, decisions, fills


def make_league(tables, notebook, name, period, n_games):
    """New league: seeded slate plus the house bots (anchored agent, raw model) deciding on it offline."""
    from evaluation.ablations import plain_policy
    rp = engine(tables)
    lg = league.create(name, rp, period, n_games)
    agent = MarketAgent(copy.deepcopy(notebook), forecaster=forecaster(), learn=False)
    return league.add_bots(lg, rp, agent.policy, plain_policy(forecaster()))


def money(x):
    return "-" if x is None or pd.isna(x) else f"{'+' if x > 0 else '−' if x < 0 else ''}${abs(x):,.2f}"


def dollars(text):
    """Escape $ so Streamlit markdown does not render the text between two amounts as LaTeX."""
    return text.replace("$", "\\$")


def fee_fmt(x):
    return "-" if x is None or pd.isna(x) else f"${x:,.2f}"


def share(x):
    return "-" if x is None or pd.isna(x) else f"{x:.0%}"


def cents(x):
    return "-" if x is None or pd.isna(x) else f"{x * 100:+.2f}¢"


def home_ticker(tables, g):
    m = tables["markets"]
    m = m[(m.game_id == g.game_id) & (m.kind == "game") & (m.team == g.home_team)]
    return m.market_ticker.iloc[0] if len(m) else None


folder, tables, names = load()
games = tables["games"]
traded = games[games.game_id.isin(tables["markets"].game_id)]
st.sidebar.title("NBA late-news agent")
st.sidebar.caption(f"data: {folder.relative_to(ROOT)} · {traded.game_id.nunique()} games with markets")
run_dirs = runs()
run = st.sidebar.selectbox("Rule notebook from run", run_dirs, format_func=lambda p: str(p.relative_to(replay.RUNS)),
                           index=0) if run_dirs else None
notebook = Notebook.load(run / "notebook.json") if run and (run / "notebook.json").exists() else Notebook()
day = st.sidebar.selectbox("Night", sorted(traded.date.unique(), reverse=True))
tonight = traded[traded.date == day]
label = {g.game_id: f"{g.away_team} @ {g.home_team}" for g in tonight.itertuples()}
gid = st.sidebar.selectbox("Game", list(label), format_func=label.get)
game = next(g for g in tonight.itertuples() if g.game_id == gid)

night, coach_tab, league_tab, briefs, learning, safety, models = st.tabs(
    ["Replayed night", "Coach: learn the market", "League: practise with play money", "Channel briefs", "Learning",
     "Safety", "Models"])
decision_times = sorted({t for t in tables["news"].loc[tables["news"].game_id == gid, "published_at"]
                         if game.tip_time - replay.NEWS_WINDOW <= t < game.tip_time} | {game.tip_time - replay.LEAD})
fmt_time = lambda t: f"{pd.Timestamp(t):%H:%M} UTC"

with night:
    st.header(f"{label[gid]} · tip {game.tip_time:%H:%M} UTC · final {game.away_pts}-{game.home_pts}")
    final_view = view_at(tables, game.tip_time)
    news = final_view.news(gid)
    st.subheader("News published before tip-off")
    st.dataframe(news[["published_at", "status", "text", "source"]], hide_index=True, width="stretch")
    m = tables["markets"]
    home_m = m[(m.game_id == gid) & (m.kind == "game") & (m.team == game.home_team)]
    if st.button("Run the agent on this night", type="primary"):
        st.session_state["night"] = run_night(tables, day, notebook)
    agent, decisions, fills = st.session_state.get("night", (None, pd.DataFrame(), pd.DataFrame()))
    if len(home_m):
        st.altair_chart(price_chart(final_view, home_m.market_ticker.iloc[0], game, news,
                                    f"{game.home_team} win price"), width="stretch")
        st.caption("orange: news published · red: tip-off (closing price)")
    if agent is not None:
        mine = [t for t in agent.traces if t["game_id"] == gid]
        for t in mine:
            with st.expander(f"decision at {pd.Timestamp(t['as_of']):%H:%M} UTC: {t['status']}", expanded=True):
                st.text(t["brief"])
                st.dataframe(pd.DataFrame(t["trace"]), hide_index=True, width="stretch")
        if len(fills):
            st.subheader("Fills tonight (settled)")
            st.dataframe(fills[["market_ticker", "side", "p_model", "price", "contracts", "fee", "close_price", "clv",
                                "pnl"]].round(3), hide_index=True)

with coach_tab:
    st.info(coach.NOTICE)
    c_when = st.select_slider("Decision time (you only see what was public then)", options=decision_times,
                              value=decision_times[0], format_func=fmt_time, key="coach_when")
    c_view = view_at(tables, c_when)
    s = coach.snapshot(forecaster(), c_view, game, names)
    key = f"{gid}|{pd.Timestamp(c_when).isoformat()}"
    paper = st.session_state.setdefault("paper", {})
    left, right = st.columns([3, 2])
    with left:
        st.subheader("What's going on")
        for line in coach.explain(s):
            st.write(line)
        if s["sides"]:
            cols = st.columns(len(s["sides"]))
            for col, side in zip(cols, s["sides"].values()):
                col.metric(f"Back {side['team']}: our estimate vs break-even", f"{side['p']:.0%}",
                           f"{side['gap'] * 100:+.1f} pts vs {side['breakeven']:.1%}")
        if len(home_m):
            shown = view_at(tables, game.tip_time) if key in paper else c_view
            st.altair_chart(price_chart(shown, home_m.market_ticker.iloc[0], game, shown.news(gid),
                                        f"{game.home_team} win price"), width="stretch")
            st.caption("price so far (orange: news). After your call the rest of the night is revealed "
                       "(red: tip-off, the closing price).")
    with right:
        st.subheader("Concepts in this game")
        for i, card in enumerate(coach.lessons(s)):
            with st.expander(card["title"], expanded=i < 2):
                st.write(card["body"])

    st.subheader("Your call")
    if key not in paper:
        if not s["sides"]:
            st.warning("No fresh market price at this time; pick another decision time.")
        else:
            with st.form(f"call-{key}"):
                choice = st.radio("What do you do?", [f"Back {s['away']}", f"Back {s['home']}", "Pass"],
                                  index=2, horizontal=True)
                stake = st.slider("Stake (paper dollars)", 5, 50, 20, 5)
                reason = st.text_input("Why? (one line, optional)")
                if st.form_submit_button("Submit my call", type="primary"):
                    if choice == "Pass":
                        paper[key] = {"filled": False, "why": "pass", "game_id": gid, "as_of": c_when,
                                      "team": None, "pick": s["pick"], "reason": reason}
                    else:
                        side = s["sides"][choice.split()[-1]]
                        paper[key] = {**coach.paper_trade(engine(tables), game, c_when, side, stake, s),
                                      "reason": reason}
                    st.rerun()
    else:
        t = paper[key]
        st.write(f"**Final score: {game.away_team} {game.away_pts:.0f}, {game.home_team} {game.home_pts:.0f}.**")
        if t.get("filled"):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"You backed {t['team']} at", f"{t['price']:.0%}", f"{t['contracts']} contracts")
            m2.metric("Closing price", f"{t['close_price']:.1%}",
                      f"{t['clv'] * 100:+.1f} cents closing-line value")
            m3.metric("Result", "won" if t["won"] else "lost")
            m4.metric("Paper P&L after fees", f"${t['pnl']:+.2f}")
        elif t["why"] == "pass":
            st.write("You passed. What backing each team would have done:")
            what_if = [coach.paper_trade(engine(tables), game, c_when, side, 20, s) for side in s["sides"].values()]
            st.dataframe(pd.DataFrame([{"back": w["team"], "price": w.get("price"), "closing price": w.get("close_price"),
                                        "won": w.get("won"), "P&L on $20": w.get("pnl")} for w in what_if]).round(3),
                         hide_index=True)
        else:
            st.warning(f"Order not filled: {t['why']}.")
        runs_key = f"agent|{day}|{gid}"
        if runs_key not in st.session_state:
            with st.spinner("Running the agent on this game..."):
                st.session_state[runs_key] = run_night(tables, day, notebook, gid)
        agent, _, agent_fills = st.session_state[runs_key]
        same = [x for x in agent.traces if pd.Timestamp(x["as_of"]) == pd.Timestamp(c_when)]
        st.write(f"**Coach's model view at this time:** "
                 + (f"back {s['pick']}." if s["pick"] else "pass (no edge after fees)."))
        if same:
            st.write(f"**What the agent did at this time:** {same[0]['status'].replace('_', ' ')}.")
            st.text(same[0]["brief"])
        if len(agent_fills):
            f = agent_fills.iloc[0]
            st.write(f"The agent's position in this game: {f.side.upper()} {f.market_ticker.rsplit('-', 1)[-1]} at "
                     f"{f.price:.0%}, closing price {f.close_price:.1%}, P&L ${f.pnl:+.2f}.")
        else:
            st.write("The agent took no position in this game.")
        if st.button("Try this decision again"):
            del paper[key]
            st.rerun()

    st.subheader("Your scoreboard")
    fb = coach.feedback(list(paper.values()))
    if fb["trades"]:
        a, b, c, d = st.columns(4)
        a.metric("Paper trades", fb["trades"])
        b.metric("P&L after fees", f"${fb['pnl']:+.2f}")
        c.metric("Closing-line value", f"${fb['clv_dollars']:+.2f}")
        d.metric("Long shots", f"{fb['long_shot_share']:.0%}")
        st.dataframe(pd.DataFrame([x for x in paper.values() if x.get("filled")])[
            ["game_id", "as_of", "team", "price", "close_price", "clv", "won", "pnl", "reason"]].round(3),
            hide_index=True, width="stretch")
    for tip in fb["tips"]:
        st.write(f"- {tip}")

with league_tab:
    st.info(league.NOTICE)
    st.write("Practise on real past Kalshi NBA prices with play money. Everyone in a league plays the same slate of "
             "decisions and sees only what was public at that moment. You are ranked on **closing-line value**: how "
             "far below the tip-off price you bought, per contract. Over many trades that is skill; one night's "
             "profit is mostly luck. House bots play the same slate so you can compare.")
    lrp = engine(tables)
    existing = league.list_leagues()
    with st.form("league-join"):
        c1, c2, c3, c4 = st.columns(4)
        lname = c1.text_input("League name", value=st.session_state.get("league_name") or
                              (existing[0] if existing else "friday-night"))
        luser = c2.text_input("Your username", value=st.session_state.get("league_user") or "")
        lperiod = c3.selectbox("Games from (new league)", list(league.PERIODS),
                               format_func={"test": "test period, Feb-Apr", "holdout": "play-offs"}.get)
        lgames = c4.number_input("Decisions per round (new league)", 3, 30, league.N_GAMES)
        if existing:
            st.caption("Existing leagues: " + ", ".join(existing))
        if st.form_submit_button("Join or create", type="primary"):
            try:
                if lname in existing:
                    lg = league.load(lname)
                else:
                    with st.spinner("Choosing the slate and running the house bots on it..."):
                        lg = make_league(tables, notebook, lname, lperiod, int(lgames))
                league.save(league.join(lg, luser))
                st.session_state.update(league_name=lname, league_user=luser, league_reveal=None)
            except ValueError as exc:
                st.error(str(exc))

    lname, luser = st.session_state.get("league_name"), st.session_state.get("league_user")
    if lname and luser and lname in league.list_leagues():
        lg = league.load(lname)
        n_slate, idx = len(lg["slate"]), league.next_index(lg, luser)
        played = n_slate if idx is None else idx
        st.subheader(f"League {lname} · {luser} · play-money bankroll ${league.balance(lg, luser):,.2f}")
        st.progress(played / n_slate, text=f"{played} of {n_slate} decisions played")
        reveal = st.session_state.get("league_reveal")

        if reveal is not None:
            t = lg["players"][luser]["picks"][str(reveal)]
            g = league._game(lrp, t["game_id"])
            st.markdown(f"#### Revealed: {g.away_team} {g.away_pts:.0f}, {g.home_team} {g.home_pts:.0f} "
                        f"(final, {g.date})")
            ticker = home_ticker(tables, g)
            close_view = view_at(tables, g.tip_time)
            q = close_view.quote(ticker) if ticker else None
            if t.get("filled"):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"You backed {t['team']} at", f"{t['price']:.0%}", f"{t['contracts']} contracts")
                m2.metric("Closing price", f"{t['close_price']:.1%}", f"{t['clv'] * 100:+.1f}¢ closing-line value")
                m3.metric("Result", "won" if t["won"] else "lost")
                m4.metric("P&L after fees", money(t["pnl"]), f"fee ${t['fee']:.2f}", delta_color="off")
            elif t["why"] == "pass":
                st.write("You passed: no money at risk, no fee."
                         + (f" {g.home_team} closed at {(q.bid + q.ask) / 2:.1%}." if q is not None else ""))
            else:
                st.warning(f"Order not filled: {t['why'].replace('_', ' ')}.")
            bots_here = {b: lg["bots"][b]["picks"].get(str(reveal), {}) for b in lg["bots"]}
            st.write(dollars("**House bots at the same moment:** " + "; ".join(
                f"{b}: " + (f"backed {p['team']} at {p['price']:.0%}, CLV {p['clv'] * 100:+.1f}¢, P&L {money(p['pnl'])}"
                            if p.get("filled") else "passed") for b, p in bots_here.items())))
            if ticker:
                st.altair_chart(price_chart(close_view, ticker, g, close_view.news(g.game_id),
                                            f"{g.home_team} win price"), width="stretch")
                st.caption(f"your decision time: {fmt_time(t['as_of'])} · orange: news · red: tip-off (closing price)")
            if st.button("Next decision" if idx is not None else "See the final standings", type="primary"):
                st.session_state["league_reveal"] = None
                st.rerun()

        elif idx is not None:
            inf = league.decision_info(forecaster(), lrp, lg, idx, names)
            g, s = inf["game"], inf["snapshot"]
            st.markdown(f"#### Decision {idx + 1} of {n_slate}: {g.away_team} @ {g.home_team}, {g.date} · "
                        f"{fmt_time(inf['now'])}, tip-off in {s['hours_to_tip']:.1f} h")
            left, right = st.columns([3, 2])
            with left:
                for line in inf["explain"]:
                    st.write(line)
                if s["sides"]:
                    cols = st.columns(len(s["sides"]))
                    for col, side in zip(cols, s["sides"].values()):
                        col.metric(f"Back {side['team']}: estimate vs break-even", f"{side['p']:.0%}",
                                   f"{side['gap'] * 100:+.1f} pts vs {side['breakeven']:.1%}")
                ticker = home_ticker(tables, g)
                if ticker:
                    st.altair_chart(price_chart(inf["view"], ticker, g, inf["news"], f"{g.home_team} win price"),
                                    width="stretch")
                    st.caption("price so far (orange: news). The rest of the night is revealed after your call.")
            with right:
                st.subheader("Concepts in this game")
                for i, card in enumerate(inf["lessons"]):
                    with st.expander(card["title"], expanded=i < 2):
                        st.write(card["body"])
            cap = league.stake_cap(lg, luser)
            with st.form(f"league-call-{lname}-{idx}"):
                options = ([f"Back {g.away_team}", f"Back {g.home_team}"] if cap >= league.STAKE_MIN else []) + ["Pass"]
                choice = st.radio("Your call", options, index=len(options) - 1, horizontal=True)
                stake = st.slider("Stake (play-money dollars)", int(league.STAKE_MIN),
                                  int(max(cap, league.STAKE_MIN + 5)), int(min(20, max(cap, league.STAKE_MIN))), 5,
                                  disabled=cap < league.STAKE_MIN)
                if st.form_submit_button("Make my call", type="primary"):
                    lg = league.load(lname)
                    pick = "pass" if choice == "Pass" else ("away" if choice.endswith(g.away_team) else "home")
                    try:
                        league.pick(lg, luser, idx, pick, float(stake), lrp, s)
                        league.save(lg)
                        st.session_state["league_reveal"] = idx
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        else:
            st.success("Round complete. Start a new league name for a fresh slate.")

        st.subheader("Leaderboard: ranked by closing-line value")
        board = league.leaderboard(lg, games)
        st.dataframe(pd.DataFrame({
            "rank": board["rank"], "name": board.name, "who": board.kind.map({"player": "player", "bot": "house bot"}),
            "decisions": board.decisions, "trades": board.trades, "pass rate": board.pass_rate.map(share),
            "mean CLV / contract": board.mean_clv.map(cents), "beat the close": board.beat_close.map(share),
            "CLV $": board.clv_dollars.map(money), "P&L": board.pnl.map(money), "fees": board.fees.map(fee_fmt),
            "skill or luck?": board.badge}), hide_index=True, width="stretch")
        st.caption(f"Ranked only with at least {league.MIN_TRADES} trades. Badge: 'skill' if the 95% day-clustered "
                   f"bootstrap interval of mean CLV is above zero, 'costs' if below, otherwise 'too early to tell'. "
                   f"P&L is shown, not ranked: over a few games it is mostly luck.")

        mine = league.summary(lg["players"][luser]["picks"], games)
        st.subheader("Your summary")
        if mine["trades"]:
            st.write(dollars(f"You beat the closing price on **{share(mine['beat_close'])}** of your {mine['trades']} "
                             f"trades (mean {cents(mine['mean_clv'])} per contract). Your fees cost "
                             f"**{fee_fmt(mine['fees'])}**; P&L after fees {money(mine['pnl'])}. You passed on "
                             f"{share(mine['pass_rate'])} of decisions. Verdict so far: **{mine['badge']}**."))
        elif mine["decisions"]:
            st.write(dollars("You have passed on every decision so far: $0, like the never-trade bot."))
        cmp = league.compare(lg, luser, games)
        if mine["decisions"]:
            st.write("On the same decisions:")
            st.dataframe(pd.DataFrame({"": cmp.name, "trades": cmp.trades, "mean CLV / contract": cmp.mean_clv.map(cents),
                                       "beat the close": cmp.beat_close.map(share), "fees": cmp.fees.map(fee_fmt),
                                       "P&L": cmp.pnl.map(money)}), hide_index=True, width="stretch")
        for tip in coach.feedback(list(lg["players"][luser]["picks"].values()))["tips"]:
            st.write(f"- {tip}")
    else:
        st.write("Enter a league name and a username to start. A new name creates a league with a fresh seeded slate.")

with briefs:
    times = decision_times
    when = st.select_slider("Decision time", options=times, value=times[-1], format_func=fmt_time)
    view = view_at(tables, when)
    now_news = view.news(gid)
    o = outlook(forecaster(), view, game, out_list(now_news))
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{game.home_team} win, before news", f"{o['p_home_before']:.0%}")
    c2.metric(f"{game.home_team} win, after news", f"{o['p_home_after']:.0%}",
              f"{(o['p_home_after'] - o['p_home_before']) * 100:+.1f} pts")
    q = view.quote(home_m.market_ticker.iloc[0]) if len(home_m) else None
    c3.metric("Market (bid-ask)", f"{q.bid:.0%}-{q.ask:.0%}" if q is not None else "none")
    b = channel_briefs(game, o, now_news, names, {"bid": q.bid, "ask": q.ask} if q is not None else None)
    cols = st.columns(2)
    for i, ch in enumerate(["platform", "media", "team", "retail"]):
        with cols[i % 2]:
            st.subheader(ch.title())
            st.write(b[ch])
    st.subheader("Player distributions (minutes and points medians; 10-90% minutes range)")
    st.dataframe(b["table"], hide_index=True, width="stretch")

with learning:
    st.subheader("Rule notebook" + (f" ({run.relative_to(replay.RUNS)})" if run else ""))
    if notebook.rules:
        st.dataframe(pd.DataFrame([{"rule": r["rule_id"], "status": r["status"], "when": json.dumps(r["when"]),
                                    "do": json.dumps(r["do"]), "valid_from": r.get("valid_from"),
                                    "gate": r.get("gate", {}).get("reason"), "why": r.get("rationale")}
                                   for r in notebook.rules]), hide_index=True, width="stretch")
    else:
        st.info("This run has no rules yet.")
    if (RESULTS / "headline_clv.png").exists():
        st.image(str(RESULTS / "headline_clv.png"))
    if (RESULTS / "ablations.md").exists():
        st.subheader("Ablations (test period)")
        st.markdown((RESULTS / "ablations.md").read_text())

with safety:
    from evaluation.scorer import policy_tests
    st.subheader("Policy tests: planted orders the risk code must block")
    st.dataframe(policy_tests(), hide_index=True)
    st.subheader("Planted fault on this game")
    fault = st.selectbox("Fault", FAULTS)
    if st.button("Run with the fault"):
        agent, _, f = run_night(tables, day, notebook, gid, plant=fault)
        for t in agent.traces:
            st.write(f"**{pd.Timestamp(t['as_of']):%H:%M} UTC: {t['status']}**")
            st.dataframe(pd.DataFrame(t["trace"]), hide_index=True, width="stretch")
        st.write(f"orders filled: {len(f)}")

with models:
    for name, title in (("calibration.csv", "Win model vs market, every test game"),
                        ("m2_heldout.csv", "Points and minutes: GRU vs baselines (held out)"),
                        ("m3_heldout.csv", "Play classifier (held out)"),
                        ("m4_heldout.csv", "Win model (held out)")):
        if (RESULTS / name).exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(RESULTS / name).round(4), hide_index=True)
