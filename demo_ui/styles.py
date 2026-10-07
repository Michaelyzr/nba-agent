"""Visual system for the presentation demo."""

CSS = r"""
<style>
:root {
  --ink: #0c1d22;
  --muted: #5d6c70;
  --line: #dbe3e1;
  --paper: #ffffff;
  --wash: #f3f6f4;
  --teal: #0d7b72;
  --teal-soft: #dff1ed;
  --lime: #cbf37a;
  --coral: #ef735d;
  --navy: #10282d;
}
.stApp { background: var(--wash); color: var(--ink); }
[data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none; }
[data-testid="stAppViewContainer"] > .main .block-container {
  max-width: 1220px; padding: 1.5rem 2.1rem 4rem;
}
html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
h1, h2, h3, p { letter-spacing: -0.02em; }
.mg-hero {
  position: relative; overflow: hidden; border-radius: 26px; padding: 30px 34px 28px;
  color: white; background:
    radial-gradient(circle at 82% 12%, rgba(203,243,122,.23), transparent 31%),
    linear-gradient(135deg, #0b252a 0%, #12373b 58%, #0d4948 100%);
  box-shadow: 0 18px 55px rgba(15,44,48,.17); margin-bottom: 18px;
}
.mg-hero:after { content:""; position:absolute; width:180px; height:180px; border:1px solid rgba(255,255,255,.12); border-radius:50%; right:-38px; bottom:-110px; }
.mg-brand { font-size: 11px; letter-spacing: .18em; text-transform: uppercase; color: #c8ded9; font-weight: 750; }
.mg-hero h1 { font-size: clamp(30px, 4vw, 51px); line-height: 1.02; margin: 22px 0 12px; max-width: 760px; letter-spacing: -.045em; }
.mg-hero p { color: #d8e7e4; font-size: 16px; max-width: 760px; margin: 0; letter-spacing: -.012em; }
.mg-pills { display:flex; flex-wrap:wrap; gap:8px; margin-top:22px; }
.mg-pill { border: 1px solid rgba(255,255,255,.18); background:rgba(255,255,255,.08); padding:7px 10px; border-radius:999px; color:#edf5f3; font-size:11px; font-weight:700; letter-spacing:.06em; }
.mg-pill.bright { color:#17312b; background:var(--lime); border-color:var(--lime); }
.mg-control-label { color:var(--muted); font-size:11px; font-weight:750; letter-spacing:.12em; text-transform:uppercase; margin: 12px 0 6px; }
.mg-gamebar { background:var(--paper); border:1px solid var(--line); border-radius:20px; padding:19px 22px; margin:14px 0 16px; display:flex; align-items:center; justify-content:space-between; gap:20px; box-shadow:0 6px 24px rgba(21,48,51,.045); }
.mg-matchup { font-size:25px; font-weight:780; letter-spacing:-.035em; }
.mg-kicker { color:var(--teal); text-transform:uppercase; font-size:10px; letter-spacing:.14em; font-weight:800; margin-bottom:5px; }
.mg-meta { color:var(--muted); font-size:13px; margin-top:5px; }
.mg-source { white-space:nowrap; background:#edf3f1; color:#35504e; border-radius:999px; padding:8px 11px; font-size:10px; font-weight:800; letter-spacing:.09em; }
.mg-grid3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin: 6px 0 14px; }
.mg-prob { background:var(--paper); border:1px solid var(--line); border-radius:18px; padding:18px; min-height:150px; box-shadow:0 5px 18px rgba(21,48,51,.035); animation:rise .35s ease both; }
.mg-prob:nth-child(2){animation-delay:.04s}.mg-prob:nth-child(3){animation-delay:.08s}
.mg-prob-label { color:var(--muted); font-size:10px; letter-spacing:.12em; text-transform:uppercase; font-weight:800; }
.mg-prob-value { font-size:38px; line-height:1; font-weight:800; letter-spacing:-.055em; margin:13px 0; color:var(--navy); }
.mg-prob-note { color:var(--muted); font-size:12px; min-height:32px; }
.mg-track { height:5px; border-radius:99px; background:#e7ecea; overflow:hidden; margin-top:14px; }
.mg-fill { height:100%; border-radius:99px; background:var(--teal); }
.mg-prob.agent { border-color:#b6d9d2; background:linear-gradient(180deg,#fff,#f2faf8); }
.mg-prob.agent .mg-fill { background:var(--coral); }
.mg-action { border-radius:18px; padding:18px 20px; display:flex; gap:15px; align-items:center; margin:8px 0 18px; border:1px solid #b8dad4; background:#e9f7f3; }
.mg-action.pass { border-color:#d9dfdd; background:#f7f9f8; }
.mg-action-mark { width:39px; height:39px; border-radius:12px; display:grid; place-items:center; background:var(--teal); color:white; font-weight:900; }
.mg-action.pass .mg-action-mark { background:#6d7c79; }
.mg-action-title { font-size:16px; font-weight:800; }
.mg-action-copy { color:var(--muted); font-size:13px; margin-top:3px; }
.mg-panel { background:var(--paper); border:1px solid var(--line); border-radius:20px; padding:21px; box-shadow:0 5px 18px rgba(21,48,51,.035); height:100%; }
.mg-panel h3 { font-size:16px; margin:0 0 12px; }
.mg-eyebrow { color:var(--teal); text-transform:uppercase; letter-spacing:.12em; font-weight:800; font-size:10px; margin-bottom:8px; }
.mg-copy { color:#425356; font-size:14px; line-height:1.58; }
.mg-news { display:flex; gap:10px; padding:9px 0; border-top:1px solid #edf0ef; align-items:flex-start; }
.mg-news:first-child { border-top:0; }
.mg-news-time { color:var(--teal); font-size:10px; font-weight:800; min-width:48px; padding-top:2px; }
.mg-news-text { font-size:12px; color:#3e5052; }
.mg-rail { display:grid; grid-template-columns:repeat(7,minmax(0,1fr)); gap:8px; margin:13px 0 20px; }
.mg-stage { position:relative; border-radius:13px; padding:12px 10px; min-height:77px; border:1px solid var(--line); background:#f8faf9; }
.mg-stage.done { background:#e8f6f2; border-color:#baddd6; }
.mg-stage.available { background:#fff8e6; border-color:#ebdcae; }
.mg-stage-dot { width:7px; height:7px; border-radius:50%; background:#b6c0be; margin-bottom:10px; }
.mg-stage.done .mg-stage-dot { background:var(--teal); box-shadow:0 0 0 4px rgba(13,123,114,.11); }
.mg-stage.available .mg-stage-dot { background:#c68a16; }
.mg-stage-name { font-size:11px; font-weight:800; line-height:1.15; }
.mg-stage-detail { color:var(--muted); font-size:9px; line-height:1.2; margin-top:5px; }
.mg-score { background:linear-gradient(135deg,#112a2f,#183d40); border-radius:22px; color:white; padding:25px 28px; display:grid; grid-template-columns:1fr auto 1fr; align-items:center; text-align:center; box-shadow:0 14px 36px rgba(14,46,49,.15); }
.mg-team { font-size:14px; font-weight:800; letter-spacing:.1em; }
.mg-score-num { font-size:45px; font-weight:850; letter-spacing:-.06em; margin-top:5px; }
.mg-clock { color:#d6e6e2; font-size:12px; font-weight:700; padding:0 24px; }
.mg-gauge { background:var(--paper); border:1px solid var(--line); border-radius:20px; padding:22px; margin-top:13px; }
.mg-gauge-top { display:flex; justify-content:space-between; align-items:end; gap:15px; }
.mg-gauge-value { font-size:38px; font-weight:850; letter-spacing:-.05em; }
.mg-bigtrack { height:11px; background:#e3e9e7; border-radius:999px; overflow:hidden; margin-top:16px; }
.mg-bigfill { height:100%; background:linear-gradient(90deg,var(--teal),#36a699); border-radius:999px; transition:width .3s ease; }
.mg-disclosure { border-left:3px solid #d4a43f; background:#fffaf0; padding:11px 14px; color:#615333; font-size:12px; border-radius:4px 12px 12px 4px; margin:12px 0; }
.mg-gradegrid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:14px 0; }
.mg-grade { background:var(--paper); border:1px solid var(--line); border-radius:16px; padding:16px; }
.mg-grade-label { color:var(--muted); text-transform:uppercase; font-size:9px; letter-spacing:.12em; font-weight:800; }
.mg-grade-value { color:var(--navy); font-size:24px; font-weight:820; margin:8px 0 3px; letter-spacing:-.04em; }
.mg-grade-note { color:var(--muted); font-size:10px; }
.mg-coach { border-radius:21px; padding:22px; background:#fff; border:1px solid #d9e2df; box-shadow:0 7px 24px rgba(21,48,51,.045); }
.mg-coach-head { display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:12px; }
.mg-nudge { border-radius:999px; padding:6px 9px; background:var(--teal-soft); color:var(--teal); text-transform:uppercase; letter-spacing:.1em; font-size:9px; font-weight:900; }
.mg-lesson { margin-top:14px; border-top:1px solid #e6ecea; padding-top:14px; }
.mg-research-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
.mg-research { background:#fff; border:1px solid var(--line); border-radius:17px; padding:16px; }
.mg-status { display:inline-block; padding:5px 7px; border-radius:999px; font-size:9px; font-weight:900; letter-spacing:.1em; color:#176c64; background:#e4f3ef; margin-bottom:10px; }
.mg-status.experimental { color:#7a5712; background:#fff1ca; }
.mg-status.evaluation-only { color:#626878; background:#eceef3; }
.mg-research-item { padding:9px 0; border-top:1px solid #edf0ef; }
.mg-research-item:first-of-type { border-top:0; }
.mg-research-name { font-weight:800; font-size:12px; }
.mg-research-desc { color:var(--muted); font-size:10px; margin-top:3px; line-height:1.35; }
.mg-footer { color:#72807e; font-size:11px; line-height:1.55; margin-top:24px; padding-top:16px; border-top:1px solid var(--line); }
[data-testid="stSegmentedControl"] { background:#fff; border:1px solid var(--line); border-radius:15px; padding:4px; }
[data-testid="stSegmentedControl"] button { min-height:38px; }
div[data-testid="stSelectbox"] > div > div { border-radius:13px; border-color:var(--line); }
button[kind="primary"] { background:var(--teal)!important; border-color:var(--teal)!important; border-radius:12px!important; }
details { border-radius:16px!important; border-color:var(--line)!important; background:#fff!important; }
@keyframes rise { from{opacity:0;transform:translateY(5px)} to{opacity:1;transform:translateY(0)} }
@media (max-width: 760px) {
  [data-testid="stAppViewContainer"] > .main .block-container { padding: .9rem .85rem 3rem; }
  .mg-hero { padding:24px 21px; border-radius:21px; }
  .mg-grid3, .mg-gradegrid, .mg-research-grid { grid-template-columns:1fr; }
  .mg-rail { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .mg-gamebar { align-items:flex-start; flex-direction:column; }
  .mg-score { padding:21px 15px; }
  .mg-score-num { font-size:35px; }
  .mg-clock { padding:0 9px; font-size:10px; }
}
</style>
"""

