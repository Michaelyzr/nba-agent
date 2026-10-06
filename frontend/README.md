# NBA Agent frontend

`app.py` is the existing seven-tab Streamlit demo. The frontend depends on the
installed backend package and has no local copies of agent or model code.
Streamlit calls the backend in the same Python process; there is no HTTP API.

Install the root `requirements.txt`, then run from the repository root:

```bash
streamlit run frontend/app.py
```

From this directory the equivalent command is `streamlit run app.py`.
Train the sample models first using the project README. API keys belong in
`backend/.env`, and the frontend does not embed them.

The independent in-play monitor starts with
`streamlit run frontend/inplay_app.py` from the repository root. See
[the in-play guide](../docs/inplay.md) for generating its logs.
