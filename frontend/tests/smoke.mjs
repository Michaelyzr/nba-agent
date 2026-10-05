// Run against the opt-in backend/tests/browser_fixture.py API, never a live DB.
// PLAYWRIGHT_MODULE can point to an existing local Playwright installation.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const output = path.resolve('.cache/qa');
const apiUrl = process.env.QA_API_URL || 'http://127.0.0.1:8001';
const frontendUrl = process.env.QA_FRONTEND_URL || 'http://127.0.0.1:3001';
await fs.mkdir(output, { recursive: true });
const profile = path.resolve(`.cache/qa-smoke-${Date.now()}`);
const context = await chromium.launchPersistentContext(profile, {
  downloadsPath: path.join(output, 'downloads'),
  channel: 'chrome', headless: true, viewport: { width: 1440, height: 1000 },
});
const page = context.pages()[0] || await context.newPage();
const errors = [];
const gameSeasons = [];
const searchModes = [];
const chatRequests = [];
page.on('request', req => {
  const url = new URL(req.url());
  if (url.pathname === '/api/v1/intelligence/games') gameSeasons.push(url.searchParams.get('season'));
  if (url.pathname === '/api/v1/analysis/chat') {
    const body = req.postDataJSON();
    searchModes.push(body.web_search); chatRequests.push(body);
  }
});
page.on('pageerror', e => errors.push(e.message));
// The intentional ambiguity response is HTTP 422; Chrome logs that expected request.
page.on('console', msg => { if (msg.type() === 'error' && !msg.text().includes('422 (Unprocessable Entity)')) errors.push(msg.text()); });

async function assertComposerVisible() {
  await page.waitForFunction(() => {
    const input = document.getElementById('analysis-question');
    const composer = input.closest('.chat-composer');
    const rect = composer.getBoundingClientRect();
    return !input.disabled && rect.top >= 0 && rect.bottom <= innerHeight;
  }, {timeout:3000});
}

try {
  const reset = await page.request.post(`${apiUrl}/__qa/reset`);
  assert.equal(reset.ok(), true, 'Start the opt-in synthetic fixture server before this test');
  await page.goto(frontendUrl);
  await page.getByRole('heading', { name: '比赛预测', exact: true }).waitFor();
  await page.getByRole('heading', { name: '先建立比赛预测' }).waitFor();
  assert.equal(await page.getByLabel('选择赛季').count(),0,'Games has no season selector');
  assert.match(await page.title(), /NBA Intelligence/);
  assert.match(await page.locator('body').innerText(), /Test Team 1/);
  await page.getByRole('button', { name: '运行 Agent ↗' }).click();
  await page.getByRole('heading', { name: '模型概率', exact: true }).waitFor();
  await page.getByRole('heading', { name: '伤病影响量化', exact: true }).waitFor();
  assert.match(await page.locator('body').innerText(), /确认缺阵/);
  assert.equal(await page.getByRole('button', { name: /模拟交易|模型回测|记录模拟仓位/ }).count(),0);
  await page.getByRole('button', { name: /模型分析/ }).click();
  assert.equal(await page.getByLabel('选择赛季').isVisible(),true,'Analysis keeps the season selector');
  assert.equal(await page.getByLabel('启用网页搜索').isChecked(),true);
  await page.getByLabel('分析球队').selectOption({label:'T1 · Test Team 1'});
  await page.getByRole('button', { name: '生成球队报告 ↗' }).click();
  await page.getByText('已完成', {exact:true}).waitFor();
  assert.match(await page.locator('.chat-history').innerText(), /Test Team 1 分析报告/);
  assert.match(await page.locator('.chat-history').innerText(), /合成测试段落 45/,'Fixture exercises a long report');
  await assertComposerVisible();
  assert.equal(await page.getByLabel('分析问题').evaluate(el=>document.activeElement===el),true);
  assert.match(await page.locator('.chat-composer').innerText(),/继续追问/);
  await page.getByText('网页检索完成 · 1 个引用来源',{exact:true}).waitFor();
  assert.equal(await page.locator('.web-citation').getAttribute('href'),'https://www.nba.com/stats');
  assert.equal(await page.locator('.chat-web-sources a').getAttribute('rel'),'noopener noreferrer');
  await page.screenshot({path:path.join(output,'analysis-web-search-desktop.png'),fullPage:true});
  await page.screenshot({path:path.join(output,'chat-composer-desktop.png')});
  for (const viewport of [{width:390,height:844},{width:760,height:900}]) {
    await page.setViewportSize(viewport);
    await assertComposerVisible();
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    await page.screenshot({path:path.join(output,`chat-composer-${viewport.width}.png`)});
  }
  await page.screenshot({path:path.join(output,'analysis-web-search-mobile.png'),fullPage:true});
  await page.setViewportSize({width:1440,height:1000});
  await page.getByLabel('分析问题').fill('解释人员配置和样本边界');
  await page.getByRole('button', { name: '发送问题 ↗' }).click();
  await page.getByText('已完成', {exact:true}).nth(1).waitFor();
  await assertComposerVisible();
  assert.equal(await page.locator('.chat-message.assistant').count(),2);
  assert.deepEqual(chatRequests[1].messages.map(m=>m.role),['user','assistant','user']);
  assert.match(chatRequests[1].messages[1].content,/Test Team 1 分析报告/);
  assert.equal(chatRequests[1].messages[2].content,'解释人员配置和样本边界');
  const downloadPromise=page.waitForEvent('download');
  await page.getByRole('button', { name: '导出 Markdown 报告 ↓' }).last().click();
  const download=await downloadPromise;
  await download.saveAs(path.join(output,'analysis-report.md'));
  assert.match(await fs.readFile(path.join(output,'analysis-report.md'),'utf8'),/数据来源/);
  assert.match(await fs.readFile(path.join(output,'analysis-report.md'),'utf8'),/\[W1\].*https:\/\/www.nba.com\/stats/);
  await page.getByLabel('启用网页搜索').uncheck();
  await page.getByRole('button', { name: /比赛预测/ }).click();
  await page.getByRole('button', { name: /模型分析/ }).click();
  assert.equal(await page.locator('.chat-message.assistant').count(),2,'History survives tab navigation');
  await page.getByLabel('搜索分析球员').fill('101');
  const playerOption=await page.getByLabel('分析球员', {exact:true}).locator('option').filter({hasText:'QA Player 101'}).getAttribute('value');
  await page.getByLabel('分析球员', {exact:true}).selectOption(playerOption);
  assert.equal(await page.locator('.chat-message').count(),0,'Scope change starts a fresh conversation');
  console.log('QA: player scope selected');
  await page.getByRole('button', { name: '生成球员报告 ↗' }).click();
  await page.getByText('已完成', {exact:true}).waitFor();
  assert.match(await page.locator('.chat-history').innerText(),/QA Player 101 分析报告/);
  assert.equal(await page.locator('.chat-web-sources').count(),0,'Local-only answers have no web sources');
  await page.locator('.chat-sources summary').click();
  await page.screenshot({path:path.join(output,'analysis-desktop.png'),fullPage:true});
  for(const viewport of [{width:390,height:844},{width:760,height:900}]) {
    await page.setViewportSize(viewport);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    await page.screenshot({path:path.join(output,`analysis-${viewport.width}.png`),fullPage:true});
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.getByLabel('分析问题').fill('测试停止');
  await page.getByRole('button', { name: '发送问题 ↗' }).click();
  await page.getByRole('button', { name: '停止生成' }).click();
  await page.getByText('已停止 · 部分结果').waitFor();
  await page.getByLabel('分析问题').fill('测试模型失败');
  await page.getByRole('button', { name: '发送问题 ↗' }).click();
  await page.getByRole('alert').filter({hasText:'模型暂时不可用'}).waitFor();
  await page.getByRole('button', { name: '新建对话', exact:true }).click();
  assert.equal(await page.locator('.chat-message').count(),0);
  await page.getByLabel('分析球队').selectOption({label:'T2 · Test Team 2'});
  await page.getByLabel('选择赛季').selectOption('2025-26');
  await page.getByLabel('分析问题').fill('QA Player 101在26-27赛季的表现');
  await page.getByRole('button', {name:'发送问题 ↗'}).click();
  await page.getByText('已完成',{exact:true}).waitFor();
  assert.equal(await page.getByLabel('分析球队').locator('option:checked').innerText(),'T1 · Test Team 1');
  assert.equal(await page.getByLabel('分析球员',{exact:true}).inputValue(),playerOption);
  assert.equal(await page.getByLabel('选择赛季').inputValue(),'2026-27');
  assert.match(await page.locator('.resolved-scope').innerText(),/QA Player 101.*2026-27.*已按问题更新范围/);
  await page.getByLabel('分析问题').fill('那他25-26季后赛呢？');
  await page.getByRole('button', {name:'发送问题 ↗'}).click();
  await page.getByText('已完成',{exact:true}).nth(1).waitFor();
  assert.equal(await page.getByLabel('分析比赛类型').inputValue(),'Playoffs');
  assert.equal(await page.getByLabel('选择赛季').inputValue(),'2025-26');
  assert.equal(await page.locator('.chat-message.assistant').count(),2,'Automatic scope updates preserve the conversation and stream');
  await page.screenshot({path:path.join(output,'intent-routing-desktop.png'),fullPage:true});
  await page.getByLabel('分析问题').fill('分析威廉姆斯的表现');
  await page.getByRole('button', {name:'发送问题 ↗'}).click();
  await page.getByText('需要澄清',{exact:true}).waitFor();
  assert.match(await page.locator('.chat-history').innerText(),/杰伦还是杰林/);
  await page.getByRole('button', {name:'新建对话',exact:true}).click();
  await page.getByRole('button', { name: /数据与设置/ }).click();
  await page.getByRole('heading', { name: '模型与风险阈值' }).waitFor();
  assert.equal(await page.getByLabel('选择赛季').count(),0,'Data settings has no season selector');
  assert.match(await page.locator('body').innerText(),/同步目标 2025-26/);
  const statsRequests=[];
  await page.route('**/api/v1/nba/sync/player-game-stats?*',async route=>{
    const url=new URL(route.request().url());
    statsRequests.push({season:url.searchParams.get('season'),type:url.searchParams.get('season_type')});
    const empty=url.searchParams.get('season_type')==='Pre Season';
    await route.fulfill({json:{data:{season:'2025-26',season_type:url.searchParams.get('season_type'),rows_received:empty?0:1,created:empty?0:1,updated:0,skipped:0}}});
  });
  for(const [type,label] of [['Regular Season','常规赛'],['Playoffs','季后赛'],['Pre Season','季前赛']]) {
    await page.getByLabel('球员统计同步比赛类型').selectOption(type);
    await page.getByRole('button',{name:`同步${label}统计`}).click();
    await page.getByRole('status').filter({hasText:type==='Pre Season'?'来源未返回球员统计':'球员统计同步完成'}).waitFor();
  }
  assert.deepEqual(statsRequests,[{season:'2025-26',type:'Regular Season'},{season:'2025-26',type:'Playoffs'},{season:'2025-26',type:'Pre Season'}]);
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:path.join(output,'data-sync-mobile.png'),fullPage:true});
  await page.setViewportSize({width:1440,height:1000});

  await page.getByRole('button', { name: /新闻与伤病/ }).click();
  await page.getByRole('heading', { name: '事件审核' }).waitFor();
  assert.equal(await page.getByLabel('选择赛季').count(),0,'News has no season selector');
  let collectedSeason;
  await page.route('**/api/v1/evidence/collect?*',async route=>{
    collectedSeason = new URL(route.request().url()).searchParams.get('season');
    await route.fulfill({json:{data:{}}});
  });
  await page.getByRole('button',{name:'立即采集 ↗'}).click();
  await page.getByRole('status').filter({hasText:'可用来源已采集'}).waitFor();
  assert.equal(collectedSeason,'2026-27','News uses the configured current season, independently of analysis');
  assert.match(await page.locator('body').innerText(), /HTTP 403/);
  await page.getByRole('button', { name: '批准事件' }).click();
  await page.getByRole('status').filter({ hasText: '事件已批准' }).waitFor();
  await page.getByLabel('筛选事件审核状态').selectOption('approved');
  await page.getByRole('button', { name: '查看归档原文' }).first().click();
  await page.getByRole('status').filter({ hasText: '已读取归档原文' }).waitFor();
  await page.getByRole('button', { name: '训练并验证伤病模型' }).click();
  await page.getByRole('status').filter({ hasText: '可用样本 0 场' }).waitFor();
  await page.screenshot({ path: path.join(output, 'news-evidence-desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth),false);
  await page.screenshot({ path: path.join(output, 'news-evidence-mobile.png'), fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByRole('button', { name: /比赛预测/ }).click();
  await page.getByRole('heading', { name: '模型概率' }).waitFor();
  assert.equal(await page.getByLabel('选择赛季').count(),0);
  assert.ok(gameSeasons.length>0);
  assert.ok(gameSeasons.every(season=>season==='2026-27'),'Analysis season changes never change the games query');
  await page.getByLabel('Best bid').fill('0.48');
  await page.getByLabel('Best ask').fill('0.50');
  await page.getByLabel('最优卖价深度 ($)').fill('2000');
  await page.getByRole('button', { name: '保存报价' }).click();
  await page.getByRole('status').filter({ hasText: '报价已保存' }).waitFor();
  await page.getByRole('button', { name: '运行 Agent ↗' }).click();
  await page.getByRole('status').filter({ hasText: 'Agent 已完成' }).waitFor();
  assert.equal(await page.getByRole('cell', { name: '确认缺阵', exact: true }).count(),0);
  assert.equal(await page.getByRole('cell', { name: '20.00 – 20.00', exact: true }).count(),1);
  assert.match(await page.locator('body').innerText(), /手动报价仅供比较，不触发研究信号/);
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: path.join(output, 'desktop.png'), fullPage: true });
  for (const viewport of [{ width: 390, height: 844 }, { width: 760, height: 900 }]) {
    await page.setViewportSize(viewport);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    assert.equal(overflow, false, `Horizontal overflow at ${viewport.width}px`);
    await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: path.join(output, `mobile-${viewport.width}.png`), fullPage: true });
  }
  assert.deepEqual(errors, []);
  assert.ok(searchModes.includes(true) && searchModes.includes(false),'Both search-enabled and local-only requests are exercised');
  console.log('PASS: page identity, real API interactions, analysis-only season selector, independent game/news season, historical stats sync, question scope override, pronoun followups, ambiguity clarification, team/player chat, reports, export, cancellation, model errors, removed modules, manual quote guard, 1440/760/390 layouts, console health');
} catch (error) {
  console.error(error);
  if (!page.isClosed()) {
  console.error(await page.locator('body').innerText());
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true });
  }
  throw error;
} finally {
  await context.close();
  await fs.rm(profile, { recursive: true, force: true });
}
