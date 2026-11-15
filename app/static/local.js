'use strict';

// Additional workspaces share the terminal's account, request and rendering lifecycle.
document.querySelector('nav').insertAdjacentHTML('beforeend', '<button data-page="performance">P&L</button>');
document.querySelector('.automation').insertAdjacentHTML('beforeend', `
  <section class="manual-ticket"><h2>Manual paper trade</h2><p id="manual-pair" class="fine"></p>
  <form id="manual-form"><label for="manual-side">Spread direction</label><select id="manual-side"><option value="long">Long: buy first / sell second</option><option value="short">Short: sell first / buy second</option></select>
  <label for="manual-budget">Gross allocation (INR)</label><input id="manual-budget" type="number" min="1000" max="500000" step="1000" value="100000" required>
  <button id="manual-open" class="primary">Open both paper legs</button></form><p class="fine">Uses the selected qualified pair and its fitted hedge ratio. Close manually from Positions.</p></section>`);
document.querySelector('#page-models').insertAdjacentHTML('beforeend', `
  <section class="method"><h2>Models used in this application</h2><div class="table-wrap"><table><thead><tr><th>MODEL / METHOD</th><th>PURPOSE</th><th>SETTINGS</th></tr></thead><tbody>
  <tr><td>Random Forest classifier</td><td>Five-minute net-profit entry score</td><td>180 trees · depth 5 · minimum leaf 10</td></tr>
  <tr><td>Paper outcome Random Forest</td><td>Learn from completed manual and automatic trades</td><td>120 trees · depth 4 · minimum leaf 4 · 40+ trades</td></tr>
  <tr><td>OLS + Engle–Granger cointegration</td><td>Hedge ratio and statistical pair screening</td><td>Return correlation ≥ 0.40 · FDR &lt; 0.05</td></tr>
  <tr><td>Daily research Random Forest</td><td>Legacy daily spread-contraction research</td><td>160 trees · depth 5 · separate from terminal automation</td></tr>
  </tbody></table></div><p>Classifier inputs: spread Z-score, absolute Z-score, normalized spread change, and spread volatility. Scores are uncalibrated estimates. The optional paper-trained filter stays disabled until trained and explicitly enabled after validation.</p><p id="candidate-universe"></p></section>
  <section class="method"><h2>Learn from paper trading</h2><p>Closed manual and automatic trades from this account only. Entry features are saved before the result is known. Local and Upstox results stay separate.</p>
  <div id="paper-training-status"></div><div class="action-row"><button id="train-paper" class="primary">Train from completed trades</button><button id="paper-filter">Enable learned entry filter</button></div><div id="paper-training-report"></div>
  <p class="fine">At least 40 eligible closed trades, including wins and losses, are required. Evaluation uses chronological 60/20/20 splits and removes outcomes overlapping the next period. The optional filter requires validation AUC and balanced accuracy above 0.50; it adds a score ≥ 0.50 gate to future automatic entries. It cannot guarantee profit. Models and filter activation must be renewed after a restart.</p></section>`);
document.querySelector('#page-history .page-title').insertAdjacentHTML('afterend', `
  <div class="history-filters"><label>Trading method<select id="history-origin"><option value="all">All methods</option><option value="auto">Automatic</option><option value="manual">Manual</option></select></label><label>Pair<select id="history-pair"><option value="all">All pairs</option><option value="banks">Banking</option><option value="it">Information technology</option><option value="auto">Automobiles</option></select></label><a class="button" href="/api/transactions/export" download="paper-transactions.csv">Export every leg fill</a></div>`);
document.querySelector('#history-table').insertAdjacentHTML('afterend','<h2 class="log-heading">All transactions · entry and exit fills</h2><div id="transaction-table"></div>');
document.querySelector('#page-history').insertAdjacentHTML('afterend', `
  <section id="page-performance" class="page content-page" hidden><div class="page-title"><div><h1>Profit & loss</h1><p id="performance-account"></p></div></div><div id="performance-total" class="history-stats"></div><h2>By trading method</h2><div id="performance-methods"></div><h2 class="log-heading">By pair / sector</h2><div id="performance-pairs"></div><p class="fine">Total = realized closed-trade P&L + open-position P&L after paid entry fees and accrued borrowing. Future exit costs are charged when positions close. Fees include paid entry and exit fees; slippage is embedded in fills. All totals cover the active account, keeping local simulation and Upstox paper accounts separate.</p></section>`);

function historyMatch(row){return ($('history-origin').value==='all'||(row.origin||'manual')===$('history-origin').value)&&($('history-pair').value==='all'||row.pair===$('history-pair').value);}
function performanceTable(rows){return '<div class="table-wrap"><table><thead><tr><th>SECTION</th><th>REALIZED</th><th>OPEN NET</th><th>TOTAL P&L</th><th>FEES</th><th>BORROW</th><th>OPEN / CLOSED</th><th>WIN / LOSS</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${esc(r.label)}</td><td class="${tone(r.realized)}">${money(r.realized)}</td><td class="${tone(r.open_net)}">${money(r.open_net)}</td><td class="${tone(r.total)}">${money(r.total)}</td><td>${money(r.fees)}</td><td>${money(r.borrow)}</td><td>${r.open} / ${r.closed}</td><td>${r.wins} / ${r.losses}</td></tr>`).join('')+'</tbody></table></div>';}
function renderLocal(){
  if(typeof renderReport==='function')renderReport();
  const local=state.local, t=state.terminal, learner=state.paper_model, pair=t.qualified.find(p=>p.id===selected);
  const universe=state.pairs.map(p=>p.id).join('|');
  if($('history-pair').dataset.universe!==universe){
    const previous=$('history-pair').value;
    $('history-pair').innerHTML='<option value="all">All pairs</option>'+state.pairs.map(p=>`<option value="${esc(p.id)}">${esc(p.y)} / ${esc(p.x)} · ${esc(p.name)}</option>`).join('');
    $('history-pair').value=state.pairs.some(p=>p.id===previous)?previous:'all';
    $('history-pair').dataset.universe=universe;
  }
  $('candidate-universe').textContent=`${new Set(state.pairs.flatMap(p=>[p.y,p.x])).size} companies · ${state.pairs.length} candidate pairs: `+state.pairs.map(p=>`${p.y} / ${p.x}`).join(', ')+'. Candidates must pass screening; inclusion is not a trading recommendation.';
  $('local-controls').hidden=!local;
  if(local){$('local-clock').textContent=`${stamp(local.time)} IST · ${local.steps} candles advanced · ${local.batch_active?local.remaining+' candles remaining':local.running?'running':'paused'}`;$('local-play').textContent=local.running||local.batch_active?'Pause local market':'Run local market';}
  $('local-play').disabled=busy||t.job.running;
  for(const id of ['local-step','local-session'])$(id).disabled=busy||t.job.running||!!local?.batch_active;
  if(local?.batch_active)for(const id of ['scan','scan-models'])$(id).disabled=true;
  $('manual-pair').textContent=pair?`${pair.y} / ${pair.x}`:'Select a qualified pair first.';
  $('manual-open').disabled=busy||!pair||!!state.error||t.job.running||state.account.positions.some(p=>p.pair===selected)||!state.pairs.find(p=>p.id===selected)?.fresh;
  $('paper-training-status').textContent=`${learner.rows} / ${learner.minimum} eligible completed trades · ${learner.wins} wins · ${learner.losses} losses · ${learner.excluded} legacy trades excluded (missing entry features).`;
  $('train-paper').disabled=busy||learner.rows<learner.minimum||t.job.running;
  $('paper-filter').disabled=busy||(!learner.enabled&&(!learner.loaded||!learner.report?.eligible));
  $('paper-filter').textContent=learner.enabled?'Disable learned entry filter':'Enable learned entry filter';
  const r=learner.report;
  $('paper-training-report').innerHTML=r?`<p>Trained ${stamp(r.trained_at)} IST · ${r.train_rows} training rows · ${r.purged_rows} overlapping rows removed.<br>Validation: AUC ${num(r.validation.auc,3)}, balanced accuracy ${pct(r.validation.balanced_accuracy)}, ${r.validation.rows} trades.<br>Untouched test: AUC ${num(r.test.auc,3)}, accuracy ${pct(r.test.accuracy)}, baseline ${pct(r.test.majority_baseline)}, ${r.test.rows} trades.<br>${r.eligible?'Validation passed.':'Validation did not pass; learned entry filter unavailable.'} ${learner.loaded?'':'Restarted: retrain to load the model.'}</p>`:'<p class="fine">Trade outcomes accumulate here automatically. No model is fitted until enough completed examples exist.</p>';
  const p=state.performance,o=p.overall;
  $('performance-account').textContent=`${state.account_key} · ${local?'Synthetic local data':'Upstox market data'} · paper money only`;
  $('performance-total').innerHTML=`<div><span>OVERALL NET P&L</span><strong class="${tone(o.total)}">${money(o.total)}</strong></div><div><span>PAPER EQUITY</span><strong>${money(state.account.equity)}</strong></div><div><span>REALIZED</span><strong class="${tone(o.realized)}">${money(o.realized)}</strong></div><div><span>OPEN NET</span><strong class="${tone(o.open_net)}">${money(o.open_net)}</strong></div>`;
  $('performance-methods').innerHTML=performanceTable(p.methods);$('performance-pairs').innerHTML=p.pairs.length?performanceTable(p.pairs):'<div class="empty">Pair results appear after your first paper trade.</div>';
  const fills=state.transactions.filter(historyMatch).slice().reverse();
  $('transaction-table').innerHTML=fills.length?'<div class="table-wrap"><table><thead><tr><th>TIME (IST)</th><th>METHOD / PAIR</th><th>EVENT</th><th>SYMBOL</th><th>SIDE / QTY</th><th>FILL</th><th>VALUE / FEE</th><th>TRADE ID</th></tr></thead><tbody>'+fills.map(f=>`<tr><td>${stamp(f.time)}</td><td>${esc(f.origin)} / ${esc(f.pair)}</td><td>${esc(f.event)}</td><td>${esc(f.symbol)}</td><td>${esc(f.side)} ${f.quantity}</td><td>${num(f.price)}</td><td>${money(f.value)}<small>Fee ${money(f.fee)}</small></td><td>${esc(f.trade_id)}</td></tr>`).join('')+'</tbody></table></div>':'<div class="empty">No matching transactions.</div>';
}
$('local-play').onclick=()=>action('/api/local',{running:!(state.local.running||state.local.batch_active)},'Local market playback updated.');
$('local-step').onclick=()=>action('/api/local',{steps:1},'Advancing one completed five-minute candle.');
$('local-session').onclick=()=>action('/api/local',{steps:75},'Advancing 75 candles in the background. Risk checks apply at every step.');
$('manual-form').onsubmit=e=>{e.preventDefault();action('/api/open',{pair:selected,side:$('manual-side').value,budget:Number($('manual-budget').value),request_id:crypto.randomUUID()},'Manual paper position opened. Both legs are recorded.');};
$('train-paper').onclick=()=>action('/api/paper/train',{},'Paper outcome model trained. Review its validation and test results.');
$('paper-filter').onclick=()=>action('/api/paper/filter',{enabled:!state.paper_model.enabled},'Learned entry filter updated.');
for(const id of ['history-origin','history-pair'])$(id).onchange=render;
