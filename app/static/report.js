'use strict';
document.querySelector('nav').insertAdjacentHTML('beforeend','<button data-page="report">Daily report</button>');
document.body.insertAdjacentHTML('beforeend',`<section id="page-report" class="page content-page" hidden>
 <div class="page-title"><div><h1>ML-enhanced pairs trading · daily report</h1><p>Your five-module project workflow: discover, verify, model, evaluate and explain.</p></div><button id="run-report" class="primary">Run all five stages</button></div>
 <div id="report-job" class="info"></div><p class="fine">Daily research runs independently of the five-minute terminal. Backtest records do not change your paper account. The selected data source is used explicitly; local daily prices are generated, not exchange history.</p>
 <div id="report-content"><div class="empty">Run the workflow to produce PCA/OPTICS discovery, verified pairs, model comparisons, a delayed holdout backtest and SHAP explanations.</div></div>
</section>`);
let reportRenderKey=null,reportSelectedPair=null;
const reportLink=(r,name)=>'/api/report/artifact?run='+encodeURIComponent(r.run_id)+'&name='+encodeURIComponent(name);
const reportImage=(r,name,alt)=>`<img class="report-plot" src="${reportLink(r,name)}" alt="${esc(alt)}" loading="lazy">`;
const reportTable=(head,body)=>`<div class="table-wrap"><table><thead><tr>${head.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${body.join('')}</tbody></table></div>`;
function renderReport(){
 if(!state?.daily_report)return;
 const {job,result:r}=state.daily_report;
 $('run-report').disabled=busy||job.running||state.mode==='live'&&!state.connected;
 $('run-report').textContent=job.running?'Daily workflow running…':'Run all five stages';
 $('report-job').textContent=(job.account?job.account+' · ':'')+job.message;
 $('report-job').className=job.error?'info error':'info';
 if(!r){
  reportRenderKey=null;reportSelectedPair=null;
  $('report-content').innerHTML='<div class="empty">Run the daily workflow for this account to produce discovery, model comparisons, a holdout backtest and explanations.</div>';
  return;
 }
 if(!r.pair_models.some(p=>p.pair===reportSelectedPair))reportSelectedPair=r.pair_models[0]?.pair||null;
 const key=r.run_id+'|'+reportSelectedPair;
 if(key===reportRenderKey)return;
 reportRenderKey=key;
 const m=r.backtest.metrics, p=r.pair_models.find(p=>p.pair===reportSelectedPair);
 const candidateRows=r.candidates.map(p=>`<tr><td>${esc(p.y)} / ${esc(p.x)}</td><td>${p.cluster}</td><td>${num(p.fdr,4)}</td><td>${num(p.hurst,3)}</td><td>${num(p.half_life,2)} days</td><td>${p.verified?'VERIFIED':esc(p.reasons.join('; '))}</td></tr>`);
 const regression=p?reportTable(['REGRESSOR','INNER TRAINING VALIDATION MSE','VALIDATION MSE','TEST MSE'],p.regression.map(x=>`<tr><td>${esc(x.model)}${x.model===p.selected_regression?' · selected':''}</td><td>${num(x.inner_validation_mse,4)}</td><td>${num(x.validation_mse,4)}</td><td>${num(x.test_mse,4)}</td></tr>`)):'';
 const models=p?reportTable(['CLASSIFIER / SETTINGS','VALIDATION 3FP + FN','VALIDATION PRECISION','TEST ACCURACY / ENTRY PRECISION','LONG / SHORT THRESHOLD','OOB ERROR'],p.models.map(x=>`<tr><td>${esc(x.model)}<small>${esc(JSON.stringify(x.parameters))}</small></td><td>${x.validation.cost_errors}</td><td>${pct(x.validation.precision)}</td><td>${pct(x.test.accuracy)} / ${pct(x.test.precision)}</td><td>${x.thresholds.long>1?'disabled':num(x.thresholds.long)} / ${x.thresholds.short>1?'disabled':num(x.thresholds.short)}</td><td>${pct(x.oob_error)}</td></tr>`)):'';
 const tradeRows=r.backtest.trades.map(t=>`<tr><td>${esc(t.symbols.join(' / '))}<small>${esc(t.side)}</small></td><td>${stamp(t.signal_time)}</td><td>${stamp(t.opened)} → ${stamp(t.closed)}</td><td>${t.quantity.join(' / ')}</td><td>${num(t.score,3)}</td><td>${esc(t.exit_reason)}</td><td>${money(t.fees)} / ${money(t.borrow)}</td><td class="${tone(t.pnl)}">${money(t.pnl)}</td></tr>`);
 $('report-content').innerHTML=`
 <div class="history-stats"><div><span>DATA SOURCE</span><strong class="report-source">${esc(r.source)}</strong></div><div><span>COMPANIES / POSSIBLE PAIRS</span><strong>${r.data.stationarity.length} / ${r.possible_pairs}</strong></div><div><span>CLUSTER CANDIDATES / VERIFIED</span><strong>${r.candidates.length} / ${r.verified}</strong></div><div><span>MODELED PAIRS</span><strong>${r.modeled}</strong></div></div>
 <section class="method"><h2>1 · Data, stationarity and PCA</h2><p>${r.data.rows} aligned daily rows · ${r.data.dropped_rows} rows dropped · formation ends ${esc(r.data.formation_end)}. ADF stationarity and all five PCA factors use formation data only. First five components explain ${pct(r.data.cumulative_variance[4])}.</p>
 ${reportImage(r,'pca_explained_variance_plot.png','Cumulative explained variance of five PCA factors')}
 <details><summary>Return stationarity checks</summary>${reportTable(['SYMBOL','ADF P-VALUE','RESULT'],r.data.stationarity.map(x=>`<tr><td>${esc(x.symbol)}</td><td>${num(x.adf_p,5)}</td><td>${x.eligible?'Included':esc(x.reason)}</td></tr>`))}</details></section>
 <section class="method"><h2>2 · OPTICS discovery and statistical verification</h2><p>Noise points remain unassigned. Candidate pairs come from within discovered clusters. Verification requires cointegration FDR &lt; 0.05, Hurst &lt; 0.5 and OU half-life between 1 and 252 trading days. These are statistical estimates, not proof of future convergence.</p>
 ${reportImage(r,'optics_cluster_map.png','OPTICS stock clusters and noise points')}${candidateRows.length?reportTable(['PAIR','CLUSTER','FDR','HURST','HALF-LIFE','RESULT'],candidateRows):'<p>No cluster candidates passed discovery.</p>'}</section>
 <section class="method"><h2>3–4 · Two-stage modeling and cost-sensitive comparison</h2>
 <label for="report-pair">Inspect a fitted pair</label><select id="report-pair">${r.pair_models.map(x=>`<option value="${esc(x.pair)}" ${x.pair===reportSelectedPair?'selected':''}>${esc(x.y)} / ${esc(x.x)}</option>`).join('')}</select>
 ${p?`<p>Selected ensemble: <b>${esc(p.selected.model)}</b>, ${esc(JSON.stringify(p.selected.parameters))}. Training / validation / test rows: ${p.train_rows} / ${p.validation_rows} / ${p.test_rows}. Six-session purge. Selection uses validation, never test results.</p><p>Latest daily action: <b>${esc(p.latest?.action||'Unavailable')}</b> · score ${num(p.latest?.score,3)} · rolling Ridge beta ${num(p.latest?.beta,4)} · next-spread forecast ${num(p.latest?.forecast,3)} · ${p.latest?.high_volatility?'high':'low'} predicted volatility.</p>
 ${regression}<p>Lasso retained: ${esc(p.retained_features.join(', '))}. ${p.lasso_fallback?'No nonzero coefficients: explicitly retaining the original features.':''} Regression training inputs use chronological out-of-fold forecasts.</p>${models}
 <details><summary>Model fitting diagnostics and Lasso coefficients</summary><pre>${esc(JSON.stringify({coefficients:p.lasso_coefficients,diagnostics:p.errors},null,2))}</pre></details>`:'<p>No verified pair had sufficient model training data.</p>'}
 <p>False-positive cost is 3× false-negative cost. Class +1 = profitable reversion proxy, 0 = noise, −1 = divergence/breakdown proxy. Long/Short direction comes from spread sign; these labels do not claim to identify a permanent structural break.</p>
 ${reportImage(r,'bias_variance_tuning_curve.png','Training and validation loss across random forest depths')}
 ${reportImage(r,'decision_boundaries_plot.png','Linear and RBF SVM two-feature probability slices')}
 ${reportImage(r,'confusion_matrix_comparison.png','Holdout confusion matrices for standard and weighted training')}</section>
 <section class="method"><h2>5 · Delayed daily holdout backtest</h2><p>${esc(r.backtest.execution)}. ${esc(r.backtest.sizing)}. Fees 3 bps + half-spread 2 bps + slippage 5 bps per fill; borrowing 10% annually. End-of-test liquidation is included.</p>
 <div class="history-stats"><div><span>NET P&L</span><strong class="${tone(m.net_pnl)}">${money(m.net_pnl)}</strong></div><div><span>ROI</span><strong>${pct(m.roi)}</strong></div><div><span>SHARPE / SORTINO</span><strong>${num(m.sharpe)} / ${num(m.sortino)}</strong></div><div><span>MAX DRAWDOWN</span><strong>${pct(m.max_drawdown)}</strong></div><div><span>WINS / LOSSES / FLAT</span><strong>${m.wins} / ${m.losses} / ${m.flat}</strong></div></div>
 <p>Break-even all-in transaction cost: ${num(m.breakeven_all_in_bps)} bps per raw traded notional after borrowing. Annualization uses 252 daily sessions and zero risk-free rate; ${m.daily_observations} holdout observations. Undefined ratios display “—”.</p>
 ${reportImage(r,'equity_curve_and_drawdown.png','Daily holdout equity, benchmark and drawdown')}
 ${reportTable(['COMMISSION BPS','NET P&L','ROI','TRADES'],r.backtest.sensitivity.map(x=>`<tr><td>${x.commission_bps}</td><td>${money(x.net_pnl)}</td><td>${pct(x.roi)}</td><td>${x.trades}</td></tr>`))}
 <h2 class="log-heading">Research trade history</h2>${tradeRows.length?reportTable(['PAIR','SIGNAL (IST)','FILL → EXIT (IST)','SHARES Y / X','SCORE','EXIT','FEES / BORROW','NET P&L'],tradeRows):'<p>No model-selected trades in this holdout. No trades were fabricated.</p>'}</section>
 <section class="method"><h2>SHAP · why each executed entry was scored</h2><p>Actual permutation SHAP for the selected model, using training-only background rows. Contributions explain entry-score probability, not rupee profit or causal economic effects. Every displayed row passes base + contributions = model score.</p>
 ${reportImage(r,'shap_summary_dependence_plots.png','SHAP summary and dependence plot for executed trades')}
 ${r.explanations.map(e=>`<details><summary>${esc(e.pair)} · ${stamp(e.time)} · score ${num(e.score,4)}</summary><p>Base ${num(e.base_value,4)} + feature contributions = ${num(e.reconstructed,4)}</p>${reportTable(['FEATURE','INPUT','SCORE CONTRIBUTION'],Object.entries(e.contributions).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${num(e.features[k],5)}</td><td>${num(v,5)}</td></tr>`))}</details>`).join('')||'<p>No executed entries to explain.</p>'}</section>
 <section class="method"><h2>Module outputs and reproducibility</h2><p>Run ${esc(r.run_id)} · ${stamp(r.created_at)} IST. Up to ${r.capacity} verified pairs are modeled, ranked using formation-period FDR only.</p>
 <div class="report-downloads">${[...r.artifacts,'report.json'].map(name=>`<a href="${reportLink(r,name)}" download>${esc(name)}</a>`).join('')}</div>
 <details><summary>Assumptions and skipped fits</summary><pre>${esc(JSON.stringify({assumptions:r.assumptions,failures:r.failures},null,2))}</pre></details></section>`;
 if($('report-pair'))$('report-pair').onchange=e=>{reportSelectedPair=e.target.value;renderReport();};
}
$('run-report').onclick=()=>action('/api/report/run',{},'Daily report started. Progress and outputs appear in this workspace.');
