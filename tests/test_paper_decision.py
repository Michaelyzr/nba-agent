"""Published edge rule for one paper decision, and the Trading tests tab that runs it."""
import json
import re
import subprocess
from pathlib import Path

import pytest

from agents.paper_decision import FEE_RATE, ORDER_CAP, paper_decision, trading_agent_decision


def _fee(price):
    return FEE_RATE * price * (1 - price)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "agents" / "templates" / "loop_demo.html"
SCRIPT = ROOT / "agents" / "templates" / "match_dashboard.js"
NODE = "node"


def test_fill_when_edge_clears_four_cents_and_clamps_the_order():
    kept = paper_decision(0.62, 0.50, 4, 20)
    assert kept["status"] == "PAPER FILL"
    assert kept["stake"] == 20
    assert kept["fee"] == pytest.approx(_fee(0.50))
    assert kept["edge"] == pytest.approx(0.62 - 0.50 - kept["fee"])
    assert kept["edge_cents"] == pytest.approx(10.25)
    assert "10.25¢" in kept["reason"]

    clamped = paper_decision(0.62, 0.50, 4, 80)
    assert clamped["status"] == "PAPER FILL" and clamped["stake"] == ORDER_CAP

    ask = 0.50
    tied = ask + _fee(ask) + 0.04
    exact = paper_decision(tied, ask, 4, 20)
    assert exact["status"] == "PASS"
    assert exact["edge_cents"] == pytest.approx(4)
    assert exact["stake"] is None

    no_edge = paper_decision(0.55, 0.54, 4, 20)
    assert no_edge["status"] == "PASS"
    assert no_edge["edge_cents"] == pytest.approx(-0.7388, abs=1e-3)
    assert "not above" in no_edge["reason"]


@pytest.mark.parametrize("p,ask,gap,stake,reason", [
    (None, 0.5, 4, 20, "missing"),
    (0.01, 0.5, 4, 20, "0.02 and 0.98"),
    (0.99, 0.5, 4, 20, "0.02 and 0.98"),
    (0.02, 0, 4, 20, "greater than 0 and less than 1"),
    (0.02, 1, 4, 20, "greater than 0 and less than 1"),
    (0.50, 0.40, "nope", 20, "edge gap"),
    (0.50, 0.40, 4, 0, "greater than $0"),
    (0.50, 0.40, 4, -5, "greater than $0"),
])
def test_invalid_inputs_pass_without_a_fill(p, ask, gap, stake, reason):
    result = paper_decision(p, ask, gap, stake)
    assert result["status"] == "PASS" and result["stake"] is None
    assert reason in result["reason"]
    assert result["edge"] is None


def test_bounds_of_probability_are_allowed():
    low = paper_decision(0.02, 0.01, 0, 20)
    high = paper_decision(0.98, 0.90, 4, 50)
    assert low["status"] == "PAPER FILL" and low["stake"] == 20
    assert high["edge_cents"] == pytest.approx((0.98 - 0.90 - _fee(0.90)) * 100)


def test_trading_tests_tab_is_static_and_next_to_demo():
    html = TEMPLATE.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")
    server = (ROOT / "agents" / "dashboard_server.py").read_text(encoding="utf-8")
    assert 'id="demo"' in html and 'id="trading"' in html and 'id="tradingtests"' in html
    assert html.index('id="demo"') < html.index('id="trading"')
    assert "Trading tests" in html and "Run paper decision" in html
    for phrase in (
        "tip−60", "0.07·p·(1−p)", "8 of 182", "+$7.38", "Fleiss κ ≈ 0.15",
        "157 trades", "−$25", "−$569", "+$564", "276", "−$407", "711", "−$2,475",
        "+$382", "87-game holdout: 0 trades", "AUC ≈ 0.53", "2/174",
        "κ ≈ 0.39", "+$0.83", "0/8", "55 of 182", "+$39.87",
        "−1.4¢", "−2¢", "Player-points", "gated abstention",
        "$100 per game", "$300 per day", "published edge rule, not a new live sceptic call",
        "Illustrative archive card", "already priced in", "10.25¢",
    ):
        assert phrase in html, phrase
    assert "runs/" not in html and "nba-agent-github" not in html
    assert "edge = p − ask − 0.07·ask·(1−ask)" in html
    assert "function paperDecision" in script and "function tradingAgentDecision" in script
    assert "PAPER FILL" in script and "Run paper decision by the trading agent" in html
    tail = script[script.index("function paperDecision"):]
    assert "kalshi.com" not in tail.lower() and "polymarket.com" not in tail.lower()
    assert set(re.findall(r"/api/[A-Za-z0-9_/-]+", tail)) == {"/api/trading-paper"}
    assert "llm" not in tail.lower() or "no llm" in tail.lower()
    assert "route.path in ('/', '/demo.html')" in server and "/api/trading-paper" in server
    assert "dashboard_html()" in server and "trading_agent_decision" in server


def test_trading_agent_names_missing_inputs_caps_and_anchor():
    plain = trading_agent_decision(0.62, 0.50, 4, 20)
    assert plain["status"] == "PAPER FILL" and plain["stake"] == 20
    assert plain["fee"] == pytest.approx(_fee(0.50))
    assert plain["edge_cents"] == pytest.approx(10.25)
    assert plain["forecast_source"] == "form" and plain["cap"] == "none"
    assert plain["llm"] is False and plain["sceptic"] is False and plain["order_sent"] is False
    for phrase in ("Kalshi 24h anchor is missing", "M4 news shift is missing", "Live quote is missing",
                   "Fee: 1.75¢", "10.25¢", "No sceptic", "no LLM", "no live order"):
        assert phrase in plain["reason"], phrase

    clamped = trading_agent_decision(0.62, 0.50, 4, 80)
    assert clamped["status"] == "PAPER FILL" and clamped["stake"] == ORDER_CAP
    assert clamped["cap"] == "$50 order cap" and "$50.00" in clamped["reason"]

    anchored = trading_agent_decision(0.10, 0.90, 4, 20, anchor=0.50, m4_shift=0.12, live_ask=0.50)
    assert anchored["forecast_source"] == "anchor+m4"
    assert anchored["p_used"] == pytest.approx(0.62) and anchored["ask_used"] == pytest.approx(0.50)
    assert anchored["status"] == "PAPER FILL"
    assert "M4 shift +0.1200" in anchored["reason"] and "Live quote ask 0.5000." in anchored["reason"]
    assert "missing" not in anchored["reason"]

    clipped = trading_agent_decision(None, None, 4, 20, anchor=0.99, m4_shift=0.05, live_ask=0.50)
    assert clipped["p_used"] == pytest.approx(0.98)

    only_anchor = trading_agent_decision(0.62, 0.50, 4, 20, anchor=0.40)
    assert only_anchor["forecast_source"] == "form" and only_anchor["p_used"] == pytest.approx(0.62)
    assert "M4 news shift is missing" in only_anchor["reason"]

    blocked = trading_agent_decision(0.62, 0.50, 4, 20, spent_game=100)
    assert blocked["status"] == "PASS" and blocked["stake"] is None
    assert blocked["cap"] == "$100 game cap" and "leaves no stake" in blocked["reason"]

    day = trading_agent_decision(0.62, 0.50, 4, 40, spent_day=280)
    assert day["status"] == "PAPER FILL" and day["stake"] == 20 and day["cap"] == "$300 day cap"

    passed = trading_agent_decision(0.55, 0.54, 4, 20)
    assert passed["status"] == "PASS" and "not above" in passed["reason"] and passed["cap"] == "not applied"


def test_browser_trading_agent_matches_python():
    script = SCRIPT.read_text(encoding="utf-8")
    start, end = script.index("function finite"), script.index("function fieldOr")
    cases = []
    payloads = [
        {"p": 0.62, "ask": 0.50, "gap_cents": 4, "stake": 20},
        {"p": 0.62, "ask": 0.50, "gap_cents": 4, "stake": 80},
        {"p": 0.55, "ask": 0.54, "gap_cents": 4, "stake": 20},
        {"p": 0.10, "ask": 0.90, "gap_cents": 4, "stake": 20, "anchor": 0.50, "m4_shift": 0.12, "live_ask": 0.50},
        {"p": 0.62, "ask": 0.50, "gap_cents": 4, "stake": 20, "anchor": 0.40},
        {"p": 0.62, "ask": 0.50, "gap_cents": 4, "stake": 20, "spent_game": 100},
        {"p": 0.62, "ask": 0.50, "gap_cents": 4, "stake": 40, "spent_day": 280},
        {"p": None, "ask": 0.50, "gap_cents": 4, "stake": 20},
        {"gap_cents": 4, "stake": 20, "anchor": 0.99, "m4_shift": 0.05, "live_ask": 0.50},
    ]
    for body in payloads:
        result = trading_agent_decision(
            body.get("p"), body.get("ask"), body.get("gap_cents", 4), body.get("stake", 20),
            anchor=body.get("anchor"), m4_shift=body.get("m4_shift"), live_ask=body.get("live_ask"),
            spent_game=body.get("spent_game") or 0, spent_day=body.get("spent_day") or 0)
        cases.append({"body": body, "status": result["status"], "reason": result["reason"],
                      "stake": result["stake"], "cap": result["cap"], "edge_cents": result["edge_cents"]})
    source = script[start:end] + """
const cases = CASES;
for (const row of cases) {
  const got = tradingAgentDecision(row.body);
  if (got.status !== row.status || got.reason !== row.reason || got.cap !== row.cap) {
    throw new Error(JSON.stringify({row, got}));
  }
  const stake = got.stake == null ? null : got.stake;
  if (stake !== row.stake) throw new Error('stake ' + stake + ' vs ' + row.stake);
  const cents = got.edge_cents == null ? null : got.edge_cents;
  if (cents === null ? row.edge_cents !== null : Math.abs(cents - row.edge_cents) > 1e-9) {
    throw new Error('cents ' + cents + ' vs ' + row.edge_cents);
  }
}
console.log('trading agent matched');
""".replace("CASES", json.dumps(cases))
    run = subprocess.run([NODE, "-e", source], capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "trading agent matched" in run.stdout


def test_browser_rule_matches_python():
    script = SCRIPT.read_text(encoding="utf-8")
    start, end = script.index("function finite"), script.index("function prefillPaper")
    cases = []
    for args in (
        (0.62, 0.50, 4, 20), (0.62, 0.50, 4, 80), (0.55, 0.54, 4, 20),
        (0.02, 0.01, 4, 20), (None, 0.5, 4, 20), (0.5, 0.4, 4, 0), (0.5, 0.4, 4.5, 20),
    ):
        result = paper_decision(*args)
        cases.append({"args": list(args), "status": result["status"], "reason": result["reason"],
                      "stake": result["stake"], "edge_cents": result["edge_cents"]})
    source = script[start:end] + """
const cases = CASES;
for (const row of cases) {
  const got = paperDecision(...row.args);
  if (got.status !== row.status || got.reason !== row.reason) {
    throw new Error(JSON.stringify({row, got}));
  }
  const stake = got.stake == null ? null : got.stake;
  if (stake !== row.stake) throw new Error('stake ' + stake + ' vs ' + row.stake);
  const cents = got.edgeCents == null ? null : got.edgeCents;
  if (cents === null ? row.edge_cents !== null : Math.abs(cents - row.edge_cents) > 1e-9) {
    throw new Error('cents ' + cents + ' vs ' + row.edge_cents);
  }
}
console.log('paper rule matched');
""".replace("CASES", json.dumps(cases))
    run = subprocess.run([NODE, "-e", source], capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "paper rule matched" in run.stdout


def test_dashboard_route_inlines_the_trading_tab():
    """Same assembly as dashboard_html(), without importing the live server."""
    bundled = (ROOT / "docs" / "samples" / "news_loop" / "demo.html").read_text(encoding="utf-8")
    data = re.search(r'<script type="application/json" id="loop-data">(.*?)</script>', bundled, re.S)
    assert data is not None
    templates = ROOT / "agents" / "templates"
    page = (templates / "loop_demo.html").read_text(encoding="utf-8")
    page = page.replace("__LOOP_DATA__", data[1])
    page = page.replace("__ANALYSIS_SCRIPT__", (templates / "betting_math.js").read_text(encoding="utf-8"))
    page = page.replace("__DASHBOARD_SCRIPT__", (templates / "match_dashboard.js").read_text(encoding="utf-8"))
    assert page.count("__LOOP_DATA__") == 0 and page.count("__DASHBOARD_SCRIPT__") == 0
    assert 'id="trading"' in page and "function paperDecision" in page and "Run paper decision" in page
    assert json.loads(data[1])["pregame_steps"]
    ids = re.findall(r'\bid="([^"]+)"', page)
    scripts = "\n".join(re.findall(r"<script>\s*(.*?)</script>", page, re.S))
    harness = r'''
class Element {
  constructor(){this.children=[];this.textContent='';this.value='';this.clientWidth=520;this.attrs={};this.hidden=false;this.disabled=false;this.className='';}
  setAttribute(k,v){this.attrs[k]=v;}
  appendChild(n){this.children.push(n);return n;}
  replaceChildren(...n){this.children=n;}
  click(){if(this.onclick)return this.onclick();}
  after(){}
  get firstChild(){return this.children[0];}
}
const E={};IDS.forEach(id=>E[id]=new Element());
E['loop-data'].textContent=DATA;
Object.assign(E.stake,{value:'20'});Object.assign(E.legs,{value:'2'});Object.assign(E.goal,{value:'profit'});
Object.assign(E.heldstake,{value:'20'});Object.assign(E.heldodds,{value:'2'});
Object.assign(E.alertseverity,{value:'all'});Object.assign(E.alerttype,{value:'all'});
Object.assign(E.papergap,{value:'4'});Object.assign(E.paperstake,{value:'20'});
for (const id of ['tradingtests','parlaycontrols','chartcard','paperresult']) E[id].hidden = true;
globalThis.document={getElementById:id=>E[id],createElement:()=>new Element(),createElementNS:()=>new Element()};
globalThis.window={addEventListener(){}};
globalThis.location={protocol:'file:'};
globalThis.setInterval=()=>1;globalThis.clearInterval=()=>{};
''' + scripts + r'''
if(E.tradingtests.hidden!==true)throw new Error('trading tab should start hidden');
E.trading.click();
if(E.matchview.hidden!==true||E.tradingtests.hidden!==false)throw new Error('trading tab did not open');
if(E.trading.attrs['aria-selected']!=='true'||E.demo.attrs['aria-selected']!=='false')throw new Error('tab state');
if(!E.paperp.value||!E.paperask.value)throw new Error('expected a prefill from the demo match');
E.paperp.value='0.62';E.paperask.value='0.50';E.paperstake.value='80';
E.paperrun.click();
if(E.paperstatus.textContent!=='PAPER FILL')throw new Error(E.paperstatus.textContent+' '+E.paperreason.textContent);
if(!E.paperreason.textContent.includes('$50.00')||!E.paperreason.textContent.includes('10.25¢'))throw new Error(E.paperreason.textContent);
E.paperp.value='0.55';E.paperask.value='0.54';E.paperstake.value='20';
E.paperrun.click();
if(E.paperstatus.textContent!=='PASS'||!E.paperreason.textContent.includes('not above'))throw new Error(E.paperreason.textContent);
E.demo.click();
if(E.tradingtests.hidden!==true||E.matchview.hidden!==false)throw new Error('demo tab did not return');
console.log('trading tab click path passed');
'''
    path = ROOT / "tests" / "_paper_tab_check.js"
    path.write_text("const IDS=" + json.dumps(ids) + ";const DATA=" + json.dumps(data[1]) + ";\n" + harness, encoding="utf-8")
    try:
        run = subprocess.run([NODE, str(path)], capture_output=True, text=True, timeout=20)
    finally:
        path.unlink(missing_ok=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "trading tab click path passed" in run.stdout
