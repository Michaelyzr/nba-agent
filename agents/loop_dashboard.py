"""Compact English match view shared by the main and standalone Streamlit apps."""
import json
import time

from agents.loop_demo import SAMPLE_OUTPUT
from agents.match_view import match_view


def _controller():
    import streamlit as st
    from agents.dashboard_live import LiveDashboard
    return st.cache_resource(LiveDashboard)()


def _render_view(view):
    import altair as alt
    import pandas as pd
    import streamlit as st

    score = view['score']
    st.subheader(f"{view['away_team']} @ {view['home_team']}")
    st.caption((f"{score['away_score']} – {score['home_score']} · " if score else '')+view['clock'])
    odds, action = st.columns([1.6, 1])
    with odds:
        st.markdown('**Where the odds stand**')
        st.dataframe(pd.DataFrame([{'Outcome': r['team'], 'Win chance': f"{r['probability']:.1%}" if r['probability'] is not None else '—',
                                  'Our odds': f"{r['model_odds']:.2f}" if r['model_odds'] else '—',
                                  'Polymarket': (f"{r['market_odds']:.2f}"+(' · Last seen' if r.get('price_status') == 'Last seen' else '')) if r['market_odds'] else '—',
                                  'Edge': f"{r['edge_pp']:+.1f} pp" if r['edge_pp'] is not None else '—'} for r in view['teams']]), hide_index=True)
        st.caption('Decimal odds. Market odds include estimated fees and available depth for your budget.')
    with action:
        a = view['action']
        st.caption('Simulated next step' if view['simulation'] else 'Suggested next step')
        st.subheader(a['title'])
        st.write(a['reason'])
        st.metric('Estimated sale proceeds' if a['kind'] == 'reduce' else 'Suggested spend',
                  f"${a.get('proceeds', 0) if a['kind'] == 'reduce' else a['amount']:.2f}")
        if a.get('floor') is not None:
            st.metric('Lower game-result P/L', f"${a['floor']:.2f}")
        elif a.get('ev') is not None:
            st.metric('Estimated net return', f"${a['ev']:.2f}")
        with st.expander('Why this action?'):
            if a.get('options'):
                st.write('Compare the lower payout across the two completed-game results, after costs. A price edge alone is not a hedge.')
                st.dataframe(pd.DataFrame([{'Option': r['kind'].title(), 'Held team wins': f"${r['if_held_wins']:.2f}",
                                          'Other team wins': f"${r['if_other_wins']:.2f}"} for r in a['options']]), hide_index=True)
                st.caption(a.get('note', ''))
            else:
                st.write('Compare buying either team with waiting, after fees, depth and a 2 percentage-point model stress buffer. Estimates are being validated.')
    chart, updates = st.columns([1.6, 1])
    with chart:
        st.markdown('**Win chance through the game**')
        rows = [{**r, 'Our estimate': r['probability']*100, 'Polymarket': r['market']*100 if r['market'] is not None else None} for r in view['curve']]
        if rows:
            frame = pd.DataFrame(rows).melt(id_vars=['minute', 'clock'], value_vars=['Our estimate', 'Polymarket'], var_name='Series', value_name='Chance')
            st.altair_chart(alt.Chart(frame).mark_line().encode(
                x=alt.X('minute:Q', title='Game minutes · Q1–Q4', scale=alt.Scale(domain=[0, max(48, max(r['minute'] for r in rows))]), axis=alt.Axis(values=[0, 12, 24, 36, 48])),
                y=alt.Y('Chance:Q', title='Home win chance (%)', scale=alt.Scale(domain=[0, 100])),
                color=alt.Color('Series:N', scale=alt.Scale(domain=['Our estimate', 'Polymarket'], range=['#067565', '#8056b1']), title=None),
                tooltip=['clock', 'Series', alt.Tooltip('Chance:Q', format='.1f')]).properties(height=240), use_container_width=True)
        else:
            st.info('Waiting for the first in-game forecast.')
    with updates:
        st.markdown('**Latest match updates**')
        for event in reversed(view['events'][-4:]):
            st.caption(event['clock']+(' · Simulated update' if event['synthetic'] else ''))
            st.write(event['headline'])
            st.caption(f"{view['home_team']} win chance {event['impact_pp']:+.2f} pp" if event['selected'] and event['impact_pp'] is not None else 'Unconfirmed update · no additional change')
        if not view['events']:
            st.write('No new player alerts. Score and market monitoring continue.')
        st.caption(view.get('connection_note') or ('Simulated Polymarket prices.' if view['simulation'] else 'Polymarket connected.' if view['market_connected'] else 'Looking for a matching Polymarket market.'))
    st.caption('Demo uses simulated scores, news and prices. Model estimates are still being validated.' if view['simulation'] else 'Live game updates and public market prices. Model estimates are still being validated.')
    st.markdown(f"[View Polymarket ↗]({view['market_url'] or 'https://polymarket.com/sports/nba'}) · Read-only analysis")


def render_demo(key='news_loop_sample'):
    import streamlit as st

    path = SAMPLE_OUTPUT/'result.json'
    if not path.exists():
        st.info('Generate the sample to open the replay.')
        st.code('python -m agents.loop_demo')
        return
    payload = json.loads(path.read_text())
    steps = [r for r in payload['steps'] if r['phase'] == 'inplay']
    mode = st.radio('Mode', ('Demo replay', 'Live games'), horizontal=True, key=key+'_mode')
    budget = st.number_input('Available budget (USD)', min_value=1., max_value=100000., value=100., key=key+'_budget')
    position = None
    def bind_position():
        st.session_state[key+'_position_game'] = st.session_state.get(key+'_game')
    with st.expander('My position (optional)'):
        enabled = st.checkbox('I already hold a position', key=key+'_position_enabled', on_change=bind_position)
        fields = st.columns(3)
        side = fields[0].selectbox('Outcome', ('home', 'away'), format_func=lambda s: s.title(), key=key+'_position_side', on_change=bind_position)
        shares = fields[1].number_input('Shares', min_value=1., value=100., key=key+'_position_shares', on_change=bind_position)
        cost = fields[2].number_input('Original cost (USD)', min_value=0., value=60., key=key+'_position_cost', on_change=bind_position)
        if enabled:
            if cost > shares:
                st.error('Original cost cannot exceed the position payout.')
                return
            position = {'side': side, 'shares': shares, 'cost': cost}
    index_key, seek_key, play_key = [key+s for s in ('_index', '_seek', '_playing')]
    st.session_state.setdefault(index_key, min(21, len(steps)-1))
    st.session_state.setdefault(play_key, False)

    def set_step(i):
        st.session_state[index_key] = st.session_state[seek_key] = i
        st.session_state[play_key] = False

    refresh = st.fragment(run_every='5s' if mode == 'Live games' else '2s') if hasattr(st, 'fragment') else lambda f: f

    @refresh
    def show():
        if mode == 'Live games':
            try:
                controller = _controller()
                games = controller.games()
                if not games:
                    st.info('No NBA games are listed right now. Monitoring the schedule automatically.')
                    return
                game_id = st.selectbox('Choose an NBA match', [g['game_id'] for g in games],
                                       format_func=lambda v: next(g['label'] for g in games if g['game_id'] == v), key=key+'_game')
                # Keep user-entered positions attached to the selected match.
                current_position = position
                if st.session_state.get(key+'_position_game') not in (None, game_id):
                    current_position = None
                    st.info('Re-enter your position for this match.')
                if st.session_state.get(key+'_position_game') is None:
                    st.session_state[key+'_position_game'] = game_id
                _render_view(controller.poll(game_id, budget, current_position))
            except Exception:
                st.info('Waiting for live data. Game and market updates retry automatically every 5 seconds.')
            return
        index = min(st.session_state[index_key], len(steps)-1)
        if st.session_state[play_key] and time.monotonic()-st.session_state.get(key+'_tick', 0) >= 1.5:
            index = min(index+1, len(steps)-1)
            st.session_state[index_key] = st.session_state[seek_key] = index
            st.session_state[key+'_tick'] = time.monotonic()
            if index == len(steps)-1:
                st.session_state[play_key] = False

        def toggle():
            if st.session_state[index_key] == len(steps)-1:
                st.session_state[index_key] = st.session_state[seek_key] = 0
            st.session_state[play_key] = not st.session_state[play_key]
            st.session_state[key+'_tick'] = time.monotonic()

        controls = st.columns(3)
        controls[0].button('From tip-off', key=key+'_reset', on_click=set_step, args=(0,))
        controls[1].button('Pause' if st.session_state[play_key] else 'Play', key=key+'_play', on_click=toggle, disabled=not hasattr(st, 'fragment'))
        controls[2].button('Next update', key=key+'_next', on_click=set_step, args=(min(index+1, len(steps)-1),), disabled=index == len(steps)-1)
        st.session_state.setdefault(seek_key, index)
        st.slider('Match replay progress', 0, len(steps)-1, key=seek_key, on_change=lambda: set_step(st.session_state[seek_key]))
        index = st.session_state[index_key]
        _render_view(match_view(payload, steps[:index+1], steps[index]['snapshot'], budget=budget, position=position))

    show()
