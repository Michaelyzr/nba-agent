"""Visual system for the simple consumer-facing demo."""

CSS = r"""
<style>
:root {
  --ink: #142026;
  --muted: #69767b;
  --line: #dfe6e3;
  --paper: #ffffff;
  --wash: #f5f7f6;
  --green: #0c7467;
  --green-soft: #e8f4f0;
  --amber: #a86c12;
  --amber-soft: #fff5df;
  --red: #b95449;
  --navy: #132d34;
}
.stApp { background: var(--wash); color: var(--ink); }
[data-testid="stHeader"], [data-testid="stToolbar"], footer { display:none; }
[data-testid="stAppViewContainer"] > .main .block-container {
  max-width: 960px; padding: 1.4rem 2rem 4rem;
}
html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
h1, h2, h3, p { letter-spacing:-.02em; }

.cs-header { display:flex; align-items:center; min-height:34px; gap:14px; margin-bottom:12px; }
.cs-brand { color:var(--navy); font-size:12px; font-weight:900; letter-spacing:.15em; }
.cs-ball { color:#f07859; font-size:13px; margin-right:7px; }
.cs-purpose { color:var(--muted); font-size:12px; padding-left:14px; border-left:1px solid var(--line); }
.cs-safety { margin-left:auto; color:#536461; background:#e8edeb; padding:6px 9px; border-radius:999px; font-size:9px; font-weight:850; letter-spacing:.1em; }

.cs-intro { margin:2px 0 14px; }
.cs-intro h1 { font-size:clamp(30px,5vw,44px); line-height:1.02; margin:0 0 8px; letter-spacing:-.055em; color:var(--navy); }
.cs-intro p { max-width:720px; margin:0; color:var(--muted); font-size:15px; line-height:1.5; }
.cs-section-label { color:#566762; font-size:10px; font-weight:850; letter-spacing:.13em; text-transform:uppercase; margin:20px 0 7px; }

div[data-testid="stSelectbox"] > div > div { min-height:52px; border:1px solid #ccd8d4; border-radius:14px; background:#fff; box-shadow:0 3px 12px rgba(20,46,49,.04); }
div[data-testid="stSelectbox"] [data-baseweb="select"] > div { font-size:15px; font-weight:700; }
div[data-testid="stSelectbox"] [data-baseweb="select"] * { color:var(--ink)!important; }
div[data-testid="stSelectbox"] input[role="combobox"] { color:var(--ink)!important; -webkit-text-fill-color:var(--ink)!important; }
.cs-game { display:flex; align-items:end; justify-content:space-between; gap:20px; padding:16px 2px 12px; border-bottom:1px solid var(--line); }
.cs-game > div { display:flex; align-items:center; gap:10px; font-size:21px; font-weight:850; letter-spacing:-.035em; }
.cs-game b { color:#9aa5a2; font-size:12px; font-weight:700; }
.cs-game p { color:var(--muted); margin:0; font-size:11px; }
.cs-game p.live { color:var(--green); font-weight:800; }
.cs-live-error { display:flex; gap:14px; align-items:flex-start; margin:15px 0; padding:22px; border-radius:18px; border:1px solid #ead2cd; background:#fff7f5; }
.cs-live-dot { width:11px; height:11px; flex:0 0 auto; border-radius:50%; background:var(--red); margin-top:5px; box-shadow:0 0 0 5px rgba(185,84,73,.1); }
.cs-live-error span { color:var(--red); font-size:9px; font-weight:900; letter-spacing:.12em; }
.cs-live-error h2 { margin:6px 0 5px; font-size:21px; }
.cs-live-error p { margin:0; color:var(--muted); font-size:12px; line-height:1.45; }
button[kind="primary"] { background:var(--green)!important; border-color:var(--green)!important; }
button[kind="primary"], button[kind="secondary"] { border-radius:12px!important; min-height:44px; }

.cs-take { position:relative; overflow:hidden; margin:13px 0 11px; padding:22px 26px; border-radius:22px; color:#fff; background:linear-gradient(135deg,#133038,#17484a); box-shadow:0 15px 36px rgba(18,48,52,.14); }
.cs-take:after { content:""; position:absolute; width:200px; height:200px; right:-85px; top:-90px; border-radius:50%; border:1px solid rgba(255,255,255,.1); }
.cs-take.wait { background:linear-gradient(135deg,#443418,#76541b); }
.cs-take.pass { background:linear-gradient(135deg,#27383d,#3d4b4e); }
.cs-take-eyebrow { color:#b9d7d2; font-size:10px; font-weight:850; letter-spacing:.14em; }
.cs-take-main { display:flex; align-items:end; justify-content:space-between; gap:28px; margin:9px 0 7px; }
.cs-take-word { font-size:clamp(42px,8vw,62px); font-weight:900; line-height:.95; letter-spacing:-.065em; }
.cs-take-meta { display:flex; gap:24px; padding-bottom:3px; }
.cs-take-meta div { min-width:78px; }
.cs-take-meta span { display:block; color:#b9ccc9; font-size:8px; font-weight:850; letter-spacing:.12em; }
.cs-take-meta strong { display:block; margin-top:3px; font-size:15px; }
.cs-take p { color:#e3eeec; margin:0; font-size:16px; line-height:1.45; max-width:700px; }

.cs-why { display:grid; grid-template-columns:105px 1fr; gap:20px; background:#fff; border:1px solid var(--line); border-radius:18px; padding:19px 22px; }
.cs-why h2 { margin:1px 0 0; font-size:20px; }
.cs-why ul { list-style:none; padding:0; margin:0; display:grid; grid-template-columns:1fr 1fr; gap:9px 18px; }
.cs-why li { display:flex; gap:9px; color:#3f5053; font-size:12px; line-height:1.4; }
.cs-why li span { flex:0 0 auto; width:18px; height:18px; display:grid; place-items:center; border-radius:50%; font-weight:900; font-size:11px; }
.cs-why li.positive span { color:var(--green); background:var(--green-soft); }
.cs-why li.caution span { color:var(--red); background:#f9e9e6; }

.cs-numbers { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:13px 0 5px; }
.cs-numbers > div { background:#fff; border:1px solid var(--line); border-radius:17px; padding:17px 19px; }
.cs-numbers > div.gap { border-color:#b8d7d1; background:#f1f9f6; }
.cs-numbers span { display:block; color:var(--muted); font-size:9px; font-weight:850; letter-spacing:.12em; }
.cs-numbers strong { display:block; color:var(--navy); font-size:32px; line-height:1; margin:10px 0 7px; letter-spacing:-.05em; }
.cs-numbers small { color:var(--muted); font-size:10px; }

.cs-context { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
.cs-context-row { display:flex; align-items:flex-start; gap:10px; min-height:72px; padding:14px; background:#fff; border:1px solid var(--line); border-radius:15px; }
.cs-context-row span { width:20px; height:20px; flex:0 0 auto; display:grid; place-items:center; border-radius:7px; color:var(--green); background:var(--green-soft); font-size:10px; font-weight:900; }
.cs-context-row p { margin:0; color:#415155; font-size:11px; line-height:1.42; }

[data-testid="stPills"] { margin-bottom:7px; }
[data-testid="stPills"] button { border-radius:999px!important; min-height:36px!important; padding:0 14px!important; font-size:11px!important; }
.cs-answer { display:flex; gap:13px; align-items:flex-start; background:#fff; border:1px solid #cbdad6; border-radius:17px; padding:17px 19px; box-shadow:0 5px 18px rgba(20,46,49,.035); }
.cs-answer-icon { width:34px; height:34px; flex:0 0 auto; display:grid; place-items:center; border-radius:11px; color:#fff; background:var(--green); font-size:12px; font-weight:900; }
.cs-answer span { color:var(--green); font-size:9px; font-weight:850; letter-spacing:.1em; text-transform:uppercase; }
.cs-answer p { margin:4px 0 0; color:#34474a; font-size:14px; line-height:1.5; }

details { margin-top:17px; border:1px solid var(--line)!important; border-radius:15px!important; background:#fff!important; }
details summary { font-size:12px!important; font-weight:720!important; }
.cs-path { display:grid; grid-template-columns:repeat(5,1fr); gap:7px; padding:5px 0 10px; }
.cs-path-step { position:relative; padding:13px 11px; border-radius:12px; background:#f1f5f3; min-height:72px; }
.cs-path-step:not(:last-child):after { content:"→"; position:absolute; right:-7px; top:29px; z-index:2; color:#91a09d; font-size:11px; }
.cs-path-step span { display:block; color:var(--green); font-size:8px; font-weight:850; letter-spacing:.1em; text-transform:uppercase; }
.cs-path-step strong { display:block; margin-top:7px; color:#34474a; font-size:10px; line-height:1.3; }

.cs-outcome { margin-top:18px; padding:19px 21px; border-radius:18px; background:#edf1ef; border:1px solid #dce4e1; }
.cs-outcome-label { color:#657570; font-size:9px; font-weight:850; letter-spacing:.12em; }
.cs-outcome h2 { margin:7px 0 2px; font-size:21px; }
.cs-outcome p { margin:7px 0 0; color:#50605e; font-size:12px; line-height:1.45; }
.cs-outcome p.cs-score { color:var(--ink); font-weight:750; margin-top:0; }
.cs-footer { color:#7b8784; font-size:10px; text-align:center; padding:24px 0 4px; }

@media (max-width: 720px) {
  [data-testid="stAppViewContainer"] > .main .block-container { padding:.9rem .85rem 3rem; }
  .cs-header { flex-wrap:wrap; }
  .cs-purpose { order:3; width:100%; border-left:0; padding-left:0; }
  .cs-intro h1 { font-size:34px; }
  .cs-game { align-items:flex-start; flex-direction:column; gap:5px; }
  .cs-take { padding:22px 20px; }
  .cs-take-main { align-items:flex-start; flex-direction:column; gap:17px; }
  .cs-take-meta { width:100%; }
  .cs-why { grid-template-columns:1fr; gap:9px; }
  .cs-why ul { grid-template-columns:1fr; }
  .cs-numbers, .cs-context { grid-template-columns:1fr; }
  .cs-path { grid-template-columns:1fr; }
  .cs-path-step:not(:last-child):after { content:"↓"; right:50%; top:auto; bottom:-8px; }
}
</style>
"""
