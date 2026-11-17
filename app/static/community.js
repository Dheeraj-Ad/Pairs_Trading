'use strict';
document.querySelector('nav').insertAdjacentHTML('beforeend','<button data-page="community">Community data</button>');
document.body.insertAdjacentHTML('beforeend',`<section id="page-community" class="page content-page" hidden>
 <div class="page-title"><div><h1>Community snapshots & paper demo</h1><p>Optional prices from 0xramm / Indian-Stock-Market-API alongside your existing trading source.</p></div></div>
 <div class="method"><h2>Choose the workflow</h2><p><b>Local demo:</b> synthetic candles, ML-qualified pairs and automatic paper fills. <b>Upstox:</b> authenticated quotes/candles for the same paper engine. <b>Community API:</b> Yahoo-based price snapshots for comparison; exchange trade times and candle history are not provided.</p><p id="community-trading-source"></p></div>
 <div class="method"><h2>Show automated paper transactions</h2><p>Scan the current local history, enable the normal model/risk gates, then advance up to 150 synthetic five-minute candles. Entries stop at the end; existing positions finish through the risk manager. Results appear in Trade history and P&L. No entry or profit is forced.</p>
 <div class="action-row"><button id="demo-run-community" class="primary">Run automated paper demo</button><button id="demo-stop-community">Stop demo</button><button data-page="history">View trade history</button><button data-page="performance">View P&L</button></div><p id="demo-session-status" role="status"></p></div>
 <div class="method"><h2>Community API connection</h2><form id="community-form"><label for="community-url">Deployment URL</label><input id="community-url" type="url" required maxlength="500" spellcheck="false"><label class="community-checkbox"><input id="community-polling" type="checkbox"> Poll every 60 seconds</label><button id="community-refresh" type="submit" class="primary">Apply & fetch snapshots</button></form>
 <p id="community-status" role="status"></p><p id="community-provenance" class="fine"></p><p class="fine">No broker credentials are sent to this service. The advertised endpoint uses HTTP. Request receipt time does not establish market-price freshness. Saved observations remain labeled as saved after a failure.</p>
 <p><a href="https://github.com/0xramm/Indian-Stock-Market-API" target="_blank" rel="noreferrer">Provider source and API contract</a></p></div>
 <div class="method"><h2>Pair price snapshots</h2><p>Qualification below belongs to your current trading source, not to the community feed. Community prices never enter the execution ledger or model training.</p><div id="community-table"></div>
 <label for="community-pair">Plot a pair qualified by the current trading model</label><select id="community-pair"></select><div id="community-plot"></div><p class="fine">Plots show our polling observations, not exchange candles or intraperiod highs/lows. Only current model-qualified pairs can be plotted.</p></div>
</section>`);
document.querySelector('#local-controls').insertAdjacentHTML('beforeend','<button id="demo-run" class="primary">Run automated paper demo</button><button id="demo-stop">Stop demo</button>');
let communityInitialized=false,communityPair=null;
function observationPlot(result,pair){
 const series=[pair.y,pair.x].map(symbol=>({symbol,rows:(result?.history?.[symbol]||[]).slice(-120)}));
 if(series.some(s=>s.rows.length<2))return '<div class="empty">At least two actual API observations per leg are needed. No chart data is generated.</div>';
 const values=series.flatMap(s=>s.rows.map(r=>({time:Date.parse(r.observed_at),value:100*(r.price/s.rows[0].price-1)})));
 const first=Math.min(...values.map(r=>r.time)),last=Math.max(...values.map(r=>r.time));
 const lo=Math.min(0,...values.map(r=>r.value)),hi=Math.max(.001,...values.map(r=>r.value));
 return `<svg class="community-chart" viewBox="0 0 900 240" role="img" aria-label="Observed pair price changes, not exchange candles"><text x="20" y="20" fill="#a6bbdd">Observed change: ${num(lo)}% to ${num(hi)}%</text>${series.map((s,i)=>{const points=s.rows.map(r=>`${30+840*(Date.parse(r.observed_at)-first)/Math.max(1,last-first)},${205-155*(100*(r.price/s.rows[0].price-1)-lo)/(hi-lo)}`).join(' ');return `<polyline points="${points}" fill="none" stroke="${i?'#e5ba73':'#7eaaff'}" stroke-width="2"/><text x="${20+i*220}" y="232" fill="${i?'#e5ba73':'#7eaaff'}">${esc(s.symbol)} · ${s.rows.length} observations</text>`;}).join('')}</svg>`;
}
function renderCommunity(){
 if(!state?.community)return;
 const c=state.community,r=c.result,j=state.demo_session;
 if(!communityInitialized){$('community-url').value=c.base_url;$('community-polling').checked=c.enabled;communityInitialized=true;}
 $('community-refresh').disabled=busy||c.running||c.retry_seconds>0;
 $('community-polling').disabled=busy||c.running;
 $('community-refresh').textContent=c.running?'Fetching snapshots…':c.retry_seconds>0?`Next fetch in ${c.retry_seconds}s`:'Apply & fetch snapshots';
 $('community-status').textContent=c.running?'Request in progress. Trading controls remain available.':r?.error?r.error:r?.observed_at?`Last saved response: ${stamp(r.observed_at)} IST · ${Object.keys(r.rows).length}/20 stocks · ${c.enabled?'polling enabled':'polling off'}`:'No community response has been received. Fetch snapshots to check the endpoint.';
 $('community-status').className=r?.error?'negative':'';
 $('community-provenance').textContent=`${c.exchange} · ${c.base_url} · ${c.notice}`;
 $('community-trading-source').textContent=`Active paper account: ${state.account_key}. ${state.source}. Upstox token: ${state.connected?'available':'not connected'}.`;
 for(const id of ['demo-run','demo-run-community'])$(id).disabled=busy||state.mode!=='demo'||j.running||state.terminal.job.running||state.local?.batch_active||state.account.positions.length>0;
 for(const id of ['demo-stop','demo-stop-community'])$(id).disabled=busy||!j.running;
 $('demo-session-status').textContent=`${j.account?j.account+' · ':''}${j.message} · ${j.new_trades} new trades · net ${money(j.pnl)}${state.mode!=='demo'?' · Select Local simulation in connection settings to run the demo.':''}`;
 const qualified=state.terminal.qualified;
 const models=new Map(state.terminal.screen.map(p=>[p.id,p]));
 $('community-table').innerHTML='<div class="table-wrap"><table><thead><tr><th>PAIR</th><th>Y PRICE</th><th>X PRICE</th><th>CURRENT MODEL</th><th>EXCHANGE TRADE TIME</th></tr></thead><tbody>'+state.pairs.map(p=>{const a=r?.rows?.[p.y],b=r?.rows?.[p.x],m=models.get(p.id);return `<tr><td>${esc(p.y)} / ${esc(p.x)}</td><td>${a?money(a.price):'Unavailable'}</td><td>${b?money(b.price):'Unavailable'}</td><td>${esc(state.mode==='demo'?'Synthetic model':'Upstox model')} · ${m?(m.eligible?'qualified':'rejected'):'not scanned'}</td><td>Not supplied</td></tr>`;}).join('')+'</tbody></table></div>';
 if(!qualified.some(p=>p.id===communityPair))communityPair=qualified[0]?.id||null;
 $('community-pair').innerHTML=qualified.length?qualified.map(p=>`<option value="${esc(p.id)}" ${p.id===communityPair?'selected':''}>${esc(p.y)} / ${esc(p.x)}</option>`).join(''):'<option>No qualified pairs in the current source</option>';
 const pair=qualified.find(p=>p.id===communityPair);
 $('community-plot').innerHTML=pair?observationPlot(r,pair):'<div class="empty">Scan and train your trading source to qualify pairs. No fallback pair is inserted.</div>';
}
$('community-pair').onchange=e=>{communityPair=e.target.value;renderCommunity();};
$('community-polling').onchange=()=>action('/api/community',{base_url:state.community.base_url,enabled:$('community-polling').checked},'Community polling preference updated.');
$('community-form').onsubmit=async e=>{e.preventDefault();await action('/api/community',{base_url:$('community-url').value,enabled:$('community-polling').checked,refresh:true},'Community snapshot request started.');};
for(const id of ['demo-run','demo-run-community'])$(id).onclick=()=>action('/api/demo/run',{},'Automated synthetic paper demo started. Follow its transactions in History.');
for(const id of ['demo-stop','demo-stop-community'])$(id).onclick=()=>action('/api/demo/stop',{},'Stopping demo entries; existing risk exits remain active.');
