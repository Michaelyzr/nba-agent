"""English match-and-market dashboard: streamlit run loop_demo_app.py."""
import streamlit as st
from agents.loop_dashboard import render_demo

st.set_page_config(page_title='Courtside · Match & market', layout='wide')
st.title('Courtside')
st.caption('Match updates. Clear decisions.')
render_demo()
