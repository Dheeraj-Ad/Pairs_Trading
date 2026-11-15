'use strict';
let state = null;
let selected = 'banks';
let page = 'trading';
let busy = false;
let toastTimer;
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const money = value => new Intl.NumberFormat('en-IN', {style:'currency', currency:'INR', maximumFractionDigits:2}).format(value || 0);
const num = (value, digits=2) => value === null || value === undefined ? '—' : Number(value).toFixed(digits);
const percent = value => value === null || value === undefined ? '—' : (value*100).toFixed(1)+'%';
const tone = value => value < 0 ? 'negative' : 'positive';
const dateTime = value => value ? new Date(value).toLocaleString('en-IN', {timeZone:'Asia/Kolkata', day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'})+' IST' : '—';
const current = () => state.pairs.find(p => p.id === selected);
function toast(message, error=false) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').className = error ? 'error' : '';
  $('toast').hidden = false;
  toastTimer = setTimeout(() => {$('toast').hidden = true;}, error ? 12000 : 5000);
}
function showPage(next) {
  page = next;
  document.querySelectorAll('.page').forEach(el => {el.hidden = el.id !== 'page-'+next;});
  document.querySelectorAll('.nav').forEach(el => el.classList.toggle('active', el.dataset.page === next));
  $('crumb').textContent = {trading:'Trading desk', portfolio:'Paper portfolio', ml:'ML lab', connection:'Data connection'}[next];
}
function metric(label, value, note, cls='') {
  return `<div class="metric"><div class="metric-label">${label}<span>↗</span></div><div class="metric-value ${cls}">${value}</div><div class="metric-note">${note}</div></div>`;
}
function drawChart(pair) {
  const rows = pair.history?.chart || [];
  if (!rows.length) return '<div class="chart-empty">Load completed daily candles in ML Lab<br>to view this pair’s historical spread.</div>';
  const w=600, h=250, left=38, right=14, top=16, bottom=30;
  const extent = Math.max(3, Math.ceil(Math.max(...rows.map(r=>Math.abs(r.z)))));
  const x = i => left+i/(rows.length-1)*(w-left-right);
  const y = z => top+(extent-z)/(extent*2)*(h-top-bottom);
  const points = rows.map((r,i)=>`${x(i).toFixed(1)},${y(r.z).toFixed(1)}`).join(' ');
  let svg = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Historical daily spread Z-score with entry thresholds at plus and minus two">`;
  for (const z of [-2,0,2]) svg += `<line x1="${left}" x2="${w-right}" y1="${y(z)}" y2="${y(z)}" class="${z?'threshold':'grid'}"/><text x="9" y="${y(z)+4}">${z>0?'+':''}${z}.0</text>`;
  svg += `<polygon class="area" points="${x(0)},${y(0)} ${points} ${x(rows.length-1)},${y(0)}"/><polyline class="curve" points="${points}"/>`;
  [0, Math.floor(rows.length/3), Math.floor(rows.length*2/3), rows.length-1].forEach((i,j) => {svg+=`<text x="${x(i)}" y="${h-7}" text-anchor="${j===0?'start':j===3?'end':'middle'}">${esc(rows[i].date.slice(5,10))}</text>`;});
  return svg+'</svg>';
}
function positions(account) {
  if (!account.positions.length) return '<div class="empty"><span class="empty-icon">⇄</span><strong>Your next idea starts here</strong>Open a paper position to track both legs and their combined P&L.</div>';
  return '<div class="table-wrap"><table><thead><tr><th>PAIR / DIRECTION</th><th>LEGS</th><th>GROSS EXPOSURE</th><th>UNREALIZED P&L</th><th>MARK STATUS</th><th></th></tr></thead><tbody>'+account.positions.map(p=>`<tr><td>${esc(p.legs.map(l=>l.symbol).join(' / '))}<small>${esc(p.side.toUpperCase())} SPREAD · ${dateTime(p.opened)}</small></td><td>${p.legs.map(l=>`${l.qty>0?'BUY':'SELL'} ${Math.abs(l.qty)} ${esc(l.symbol)}`).join('<br>')}</td><td>${money(p.gross)}</td><td class="${tone(p.unrealized)}">${money(p.unrealized)}<small>Includes accrued borrow; entry fee in balance</small></td><td><span class="pill ${p.fresh&&!state.error?'mint':'amber'}">${p.fresh&&!state.error?'FRESH':'STALE / ESTIMATED'}</span></td><td><button class="close-position" data-close="${esc(p.id)}" ${busy||!p.fresh||state.error?'disabled':''}>Close pair</button></td></tr>`).join('')+'</tbody></table></div>';
}
function renderResearch() {
  const pair = state.pairs.find(p=>p.id===$('ml-pair').value) || current();
  const info=pair.history, report=pair.report;
  $('history-status').textContent = info ? `${info.rows} daily observations · ${info.formation_rows} formation sessions · last close ${info.history_end}. Source: ${state.mode==='demo'?'synthetic demo':'Upstox raw candles'}.` : 'Load this pair’s history before training or opening a paper position.';
  $('train-model').disabled = busy || !info;
  $('load-history').disabled = busy;
  $('model-source').textContent = report ? (report.source==='demo'?'SYNTHETIC DEMO':'UPSTOX HISTORY') : 'NOT TRAINED';
  if (!report) {
    $('model-report').innerHTML = '<div class="empty"><span class="empty-icon">⌘</span><strong>Give your hypothesis a fair test</strong>Train the model to see held-out classification metrics,<br>sample counts, and feature importance.</div><div class="research-info"><b>Research stays separate from execution</b><p>No classifier is connected to automatic trading. Synthetic performance is a demonstration, not evidence of a market opportunity.</p></div>';
    return;
  }
  $('model-report').innerHTML = `<div class="report-metrics">${metric('Holdout accuracy',percent(report.accuracy),'Selected threshold '+num(report.threshold))}${metric('ROC AUC',num(report.auc,3),'Untouched chronological test block')}${metric('Balanced accuracy',percent(report.balanced_accuracy),'Average recall across classes')}${metric('Majority baseline',percent(report.majority_baseline),'Always predict the dominant test class')}</div><p class="fine">${report.train_rows} train / ${report.validation_rows} validation / ${report.test_rows} test samples · ${report.purge_sessions}-session purge</p><h3 class="history-panel">Feature importance</h3>${Object.entries(report.feature_importance).map(([key,value])=>`<div class="feature"><span>${esc(key.replaceAll('_',' '))}</span><meter min="0" max="1" value="${value}">${percent(value)}</meter><span>${percent(value)}</span></div>`).join('')}<p class="fine">Train ends ${esc(report.splits.train_end)} · validation ends ${esc(report.splits.validation_end)} · test ends ${esc(report.splits.test_end)}.</p><p class="fine">Trained ${dateTime(report.trained_at)}. Classification accuracy is not strategy return. Inspect transaction-level backtests before drawing conclusions.</p>`;
}
function preview() {
  if (!state) return;
  const pair = current(), budget = Number($('allocation').value), beta = pair.history?.beta;
  if (!beta || !pair.quotes.every(Boolean)) {$('ticket-preview').textContent = 'Load history and connect quotes to size both legs.'; return;}
  const units = budget / ((pair.quotes[0].price+beta*pair.quotes[1].price)*1.0005);
  const qy=Math.floor(units), qx=Math.floor(units*beta), long=$('direction').value==='long';
  $('ticket-preview').textContent = `${long?'Buy':'Sell'} ${Math.max(0,qy)} ${pair.y} · ${long?'Sell':'Buy'} ${Math.max(0,qx)} ${pair.x}. Quantities recalculated at fill.`;
}
function render() {
  if (!state) return;
  const a=state.account, pair=current();
  $('feed-pill').textContent = state.mode==='demo'?'DEMO DATA':state.pairs.every(p=>p.fresh)?'LIVE · UPSTOX':'STALE · UPSTOX';
  $('feed-pill').className = 'pill '+(state.mode==='live'&&state.pairs.every(p=>p.fresh)?'mint':'amber');
  $('banner').className = 'notice'+(state.error?' error':'');
  $('banner').textContent = state.error || (state.mode==='demo'?'Synthetic demo prices · not market data. Demo trades have a separate ledger.':`Upstox ${state.exchange} · quotes poll every 5 seconds · last successful refresh ${dateTime(state.last_poll)}. ${state.pairs.every(p=>p.fresh)?'Fresh quotes available.':'Stale or closed-market quotes: new fills are paused.'}`);
  $('exchange-label').textContent=state.exchange;
  const stale = state.error || a.positions.some(p=>!p.fresh);
  const metrics = metric('Paper equity',money(a.equity),stale?'Estimated · stale marks':`${state.mode==='demo'?'Demo':'Upstox data'} · ${state.exchange} account`)+metric('Total P&L',money(a.total_pnl),'Since account creation · costs included',tone(a.total_pnl))+metric('Available capacity',money(a.available),'1× equity gross exposure limit')+metric('Open positions',String(a.positions.length).padStart(2,'0'),money(a.gross)+' gross exposure');
  $('desk-metrics').innerHTML=metrics;
  $('portfolio-metrics').innerHTML=metrics;
  $('pairs').innerHTML=state.pairs.map(p=>`<button class="pair ${p.id===selected?'selected':''}" data-pair="${p.id}" aria-pressed="${p.id===selected}"><div class="pair-name">${esc(p.y)} <span class="muted">/</span> ${esc(p.x)}</div><div class="pair-sector">${esc(p.name)}</div><div class="pair-bottom"><span>${esc(p.signal)}</span><span class="pair-z">${num(p.zscore)} <small>Z</small></span></div></button>`).join('');
  $('chart-title').textContent=pair.y+' / '+pair.x;
  $('chart-source').textContent=state.mode==='demo'?'· Synthetic':'· Daily closes';
  $('spread-chart').innerHTML=drawChart(pair);
  $('pair-stats').innerHTML=[['Current Z-score',num(pair.zscore)],['Hedge ratio β',num(pair.history?.beta,3)],['Coint. p-value',num(pair.history?.coint_p,4)]].map(([label,value])=>`<div><div class="stat-label">${label}</div><div class="stat-value">${value}</div></div>`).join('');
  $('ticket-quotes').innerHTML=pair.quotes.map((q,i)=>`<div class="quote-row"><span><small>${i===0?'LEG A':'LEG B'}</small>${esc(i===0?pair.y:pair.x)}</span><span>${q?money(q.price):'—'}</span></div>`).join('');
  $('open-order').disabled=busy||!pair.fresh||!pair.history;
  $('position-count').textContent=a.positions.length;
  $('desk-positions').innerHTML=positions(a);
  $('portfolio-positions').innerHTML=positions(a);
  $('trade-history').innerHTML=a.trades.length?'<div class="table-wrap"><table><thead><tr><th>PAIR</th><th>DIRECTION</th><th>CLOSED</th><th>FEES / BORROW</th><th>NET P&L</th></tr></thead><tbody>'+a.trades.slice().reverse().map(t=>`<tr><td>${esc(t.entry_legs.map(l=>l.symbol).join(' / '))}</td><td>${esc(t.side)} spread</td><td>${dateTime(t.closed)}</td><td>${money(t.fees)} / ${money(t.borrow)}</td><td class="${tone(t.pnl)}">${money(t.pnl)}</td></tr>`).join('')+'</tbody></table></div>':'<div class="empty"><strong>No closed trades yet</strong>Your completed paper trades will appear here.</div>';
  $('connection-status').textContent=`Active: ${state.source} · ${state.exchange}. Token ${state.connected?'available in server memory':'not connected'}.`;
  if (!$('ml-pair').options.length) $('ml-pair').innerHTML=state.pairs.map(p=>`<option value="${p.id}">${esc(p.y)} / ${esc(p.x)}</option>`).join('');
  renderResearch(); preview();
}
async function action(route, data={}, message='Saved') {
  if (busy || !state) return;
  busy=true; render();
  document.querySelectorAll('#connection-form button').forEach(b=>b.disabled=true);
  try {
    const response=await fetch(route,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':state.csrf},body:JSON.stringify({account_key:state.account_key,...data})});
    const result=await response.json();
    if (!response.ok) throw new Error(result.error || 'Request failed.');
    state=result; toast(message);
  } catch(error) {toast(error.message,true);}
  finally {busy=false; document.querySelectorAll('#connection-form button').forEach(b=>b.disabled=false); render();}
}
async function refresh() {
  if (busy) return;
  try {
    const response=await fetch('/api/state');
    if (!response.ok) throw new Error('Server unavailable');
    const result=await response.json();
    if (!busy) {state=result;render();}
  } catch(error) {
    $('banner').className='notice error';
    $('banner').textContent='Local server unavailable. Quotes and account values are not updating.';
    $('open-order').disabled=true;
    document.querySelectorAll('.close-position').forEach(b=>b.disabled=true);
  }
}
document.addEventListener('click', event=>{
  const nav=event.target.closest('[data-page], [data-go]');
  if (nav) showPage(nav.dataset.page||nav.dataset.go);
  const pair=event.target.closest('[data-pair]');
  if(pair){selected=pair.dataset.pair;render();}
  const close=event.target.closest('[data-close]');
  if(close) action('/api/close',{position_id:close.dataset.close,request_id:crypto.randomUUID()},'Both paper legs closed.');
});
$('allocation').addEventListener('input',preview);
$('direction').addEventListener('change',preview);
$('ml-pair').addEventListener('change',renderResearch);
$('open-order').addEventListener('click',()=>action('/api/open',{pair:selected,side:$('direction').value,budget:Number($('allocation').value),request_id:crypto.randomUUID()},'Paper position opened. Both legs recorded.'));
$('load-history').addEventListener('click',()=>action('/api/history',{pair:$('ml-pair').value},'Daily history loaded and spread baseline fitted.'));
$('train-model').addEventListener('click',()=>{toast('Training and evaluating the model…');action('/api/train',{pair:$('ml-pair').value},'Model trained. Holdout report and artifact saved locally.');});
$('connection-form').addEventListener('submit',async event=>{
  event.preventDefault();
  const token=$('token').value;
  $('token').value='';
  toast('Checking the data connection…');
  await action('/api/connection',{mode:$('data-mode').value,exchange:$('exchange').value,token},'Data source and paper account updated.');
});
$('disconnect').addEventListener('click',()=>action('/api/disconnect',{},'Token cleared. Demo account active.'));
function clock(){ $('clock').textContent=new Date().toLocaleTimeString('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false})+' IST'; }
clock();setInterval(clock,1000);refresh();setInterval(refresh,5000);
