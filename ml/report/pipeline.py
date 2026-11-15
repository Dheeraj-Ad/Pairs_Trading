"""Five report stages, reproducible artifacts and explicit research provenance."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ml.report.data import prepare
from ml.report.discovery import discover
from ml.report.models import train_pair, probabilities
from ml.report.backtest import backtest
from ml.report.explain import explain_trades


def save_csv(rows,path):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame=frame.copy()
    for col in frame:
        frame[col]=frame[col].map(lambda v:json.dumps(v,allow_nan=False) if isinstance(v,(dict,list)) else v)
    frame.to_csv(path,index=isinstance(rows,pd.DataFrame))


def plot(path,draw,wide=False):
    fig=Figure(figsize=(11 if wide else 8,4),layout='constrained')
    draw(fig)
    fig.savefig(path,dpi=130,facecolor='white')


def artifacts(directory,prices,returns,factors,data,clusters,pairs,bundles,result,explanations):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    save_csv(prices,directory/'daily_prices.csv');save_csv(returns,directory/'cleaned_stationary_returns.csv')
    save_csv(factors,directory/'pca_factors.csv');save_csv(data['stationarity'],directory/'stationarity_checks.csv')
    save_csv(clusters,directory/'optics_clusters.csv');save_csv(pairs,directory/'all_candidate_tests.csv')
    save_csv([r for r in pairs if r['verified']],directory/'verified_pairs_universe.csv')
    save_csv([result['metrics']],directory/'strategy_performance_tearsheet.csv')
    save_csv(result['trades'],directory/'backtest_trades.csv');save_csv(result['equity'],directory/'equity_ledger.csv')
    save_csv(result['sensitivity'],directory/'cost_sensitivity.csv');save_csv(explanations,directory/'shap_trade_explanations.csv')
    model_rows=[{'pair':key,**row} for key,b in bundles.items() for row in b['report']['models']]
    save_csv(model_rows,directory/'model_comparison.csv')
    plot(directory/'pca_explained_variance_plot.png',lambda fig: _pca_plot(fig,data))
    plot(directory/'optics_cluster_map.png',lambda fig:_cluster_plot(fig,clusters))
    plot(directory/'equity_curve_and_drawdown.png',lambda fig:_equity_plot(fig,result),wide=True)
    plot(directory/'bias_variance_tuning_curve.png',lambda fig:_depth_plot(fig,bundles))
    plot(directory/'confusion_matrix_comparison.png',lambda fig:_confusion_plot(fig,bundles),wide=True)
    plot(directory/'decision_boundaries_plot.png',lambda fig:_boundary_plot(fig,bundles),wide=True)
    plot(directory/'shap_summary_dependence_plots.png',lambda fig:_shap_plot(fig,explanations),wide=True)
    for key,bundle in bundles.items():
        save_csv(bundle['design'],directory/(key+'-features.csv'))
        save_csv(bundle['spread'],directory/(key+'-dynamic-hedge.csv'))
        save_csv(bundle['report']['regression'],directory/(key+'-regression.csv'))
        save_csv([dict(feature=k,coefficient=v,retained=k in bundle['report']['retained_features'])
                  for k,v in bundle['report']['lasso_coefficients'].items()],directory/(key+'-lasso.csv'))
    return sorted(p.name for p in directory.iterdir() if p.suffix in ('.csv','.png'))


def _pca_plot(fig,data):
    ax=fig.subplots();ax.plot(range(1,6),data['cumulative_variance'],marker='o',color='black')
    ax.set(xlabel='Principal components',ylabel='Cumulative explained variance',ylim=(0,1),title='Formation-only PCA of daily log returns')


def _cluster_plot(fig,clusters):
    ax=fig.subplots()
    for label in sorted({r['cluster'] for r in clusters}):
        rows=[r for r in clusters if r['cluster']==label]
        ax.scatter([r['pc1'] for r in rows],[r['pc2'] for r in rows],marker='x' if label==-1 else 'o',label='Noise' if label==-1 else f'Cluster {label}')
        for r in rows:ax.annotate(r['symbol'],(r['pc1'],r['pc2']),fontsize=6)
    ax.set(title='OPTICS on normalized five-factor stock loadings',xlabel='Loading coordinate 1',ylabel='Loading coordinate 2')
    ax.legend(fontsize=6,ncol=3)


def _equity_plot(fig,result):
    top,bottom=fig.subplots(2,1,sharex=True)
    rows=result['equity'];equity=np.array([r['equity'] for r in rows]);times=pd.to_datetime([r['time'] for r in rows])
    if len(rows):
        top.plot(times,equity,label='Strategy',color='black');top.plot(times,[r['benchmark'] for r in rows],label='Equal-weight universe',color='gray')
        peak=np.maximum.accumulate(np.r_[result['metrics']['capital'],equity])[1:]
        bottom.fill_between(times,1-equity/peak,color='gray');bottom.set_ylabel('Drawdown')
    top.set(title='Untouched daily holdout, after modeled costs',ylabel='Paper INR');top.legend()


def _depth_plot(fig,bundles):
    ax=fig.subplots()
    for key,b in bundles.items():
        rows=[r for r in b['report']['models'] if r['model']=='Random forest']
        ax.plot([r['parameters']['depth'] for r in rows],[r['train_loss'] for r in rows],linestyle='--',label=key+' train')
        ax.plot([r['parameters']['depth'] for r in rows],[r['validation']['weighted_log_loss'] for r in rows],label=key+' validation')
    ax.set(xlabel='Tree depth',ylabel='Weighted log-loss',title='Training / validation tuning diagnostic')
    if bundles:ax.legend(fontsize=6)


def _confusion_plot(fig,bundles):
    axes=fig.subplots(1,2)
    for ax,key,title in zip(axes,['standard_loss_test','weighted_loss_test'],['Standard training','3x non-entry class weighting']):
        matrix=sum((np.array(b['report'][key]['confusion']) for b in bundles.values()),np.zeros((3,3),dtype=int))
        ax.imshow(matrix,cmap='Greys')
        for i in range(3):
            for j in range(3):ax.text(j,i,str(matrix[i,j]),ha='center',color='red')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=['Break','Hold','Revert'],yticklabels=['Break','Hold','Revert'],xlabel='Predicted',ylabel='Actual proxy label',title=title)


def _boundary_plot(fig,bundles):
    axes=fig.subplots(1,2)
    if not bundles:
        for ax in axes:ax.text(.1,.5,'No verified model to plot');ax.axis('off')
        return
    b=next(iter(bundles.values()));x=b['train_x'];cols=list(x.columns)
    f1,f2=cols[:2]
    a=np.linspace(x[f1].min(),x[f1].max(),50);c=np.linspace(x[f2].min(),x[f2].max(),50)
    grid_a,grid_b=np.meshgrid(a,c);grid=pd.DataFrame(np.tile(x.median().to_numpy(),(2500,1)),columns=cols)
    grid[f1]=grid_a.ravel();grid[f2]=grid_b.ravel()
    for ax,name in zip(axes,['Linear SVM','RBF SVM']):
        model=b['baselines'].get(name)
        if model is None:ax.text(.1,.5,name+' unavailable');ax.axis('off');continue
        values=probabilities(model,grid)[:,2].reshape(50,50)
        ax.contourf(grid_a,grid_b,values,levels=10,cmap='Greys')
        ax.set(title=name+' positive-class score slice',xlabel=f1,ylabel=f2)


def _shap_plot(fig,rows):
    left,right=fig.subplots(1,2)
    if not rows:
        for ax in (left,right):ax.text(.1,.5,'No executed entries: no trade SHAP values');ax.axis('off')
        return
    names=sorted({name for row in rows for name in row['contributions']})
    ranking=sorted(names,key=lambda k:np.mean([abs(r['contributions'].get(k,0)) for r in rows]),reverse=True)[:8]
    rng=np.random.default_rng(42)
    for i,name in enumerate(ranking):
        values=[r['contributions'][name] for r in rows if name in r['contributions']]
        left.scatter(values,i+rng.uniform(-.15,.15,len(values)),s=10,color='black')
    left.set(yticks=range(len(ranking)),yticklabels=ranking,xlabel='Contribution to entry score',title='Executed trades: SHAP summary')
    name=ranking[0];selected=[r for r in rows if name in r['features']]
    right.scatter([r['features'][name] for r in selected],[r['contributions'][name] for r in selected],color='black')
    right.set(xlabel=name,ylabel='SHAP contribution',title='Dependence of leading feature')


def run(prices,directory,source,progress=lambda message:None):
    progress('1/5 · Cleaning daily prices, ADF checks and five-component PCA')
    prices,returns,factors,data=prepare(prices)
    progress('2/5 · OPTICS discovery, cointegration FDR, Hurst and half-life')
    clusters,pairs=discover(prices,data)
    verified=sorted([r for r in pairs if r['verified']],key=lambda r:(r['fdr'],r['id']))
    # A transparent formation-only capacity cap limits interactive work, not a holdout ranking.
    selected=verified[:8];bundles={};failures=[]
    for index,pair in enumerate(selected):
        progress(f"3–4/5 · {index+1}/{len(selected)} {pair['y']} / {pair['x']}: regression and classifiers")
        try:bundles[pair['id']]=train_pair(prices,pair)
        except ValueError as exc:failures.append(dict(pair=pair['id'],error=str(exc)))
    progress('5/5 · Next-period execution, holdout metrics and cost sensitivity')
    result=backtest(prices,bundles)
    result['sensitivity']=[dict(commission_bps=bps,**backtest(prices,bundles,fee=bps/10000)['metrics']) for bps in [0,1,3,5,10]]
    progress('5/5 · Computing actual per-entry SHAP attributions and exporting module outputs')
    explanations=explain_trades(bundles,result['trades'])
    files=artifacts(directory,prices,returns,factors,data,clusters,pairs,bundles,result,explanations)
    report=dict(version=1,source=source,timeframe='daily',data=data,clusters=clusters,candidates=pairs,
        possible_pairs=len(prices.columns)*(len(prices.columns)-1)//2,verified=len(verified),modeled=len(bundles),
        capacity=8,failures=failures,pair_models=[b['report'] for b in bundles.values()],
        backtest=result,explanations=explanations,artifacts=files,
        assumptions=['Synthetic mode is generated data, not historical exchange prices.',
            'Report class -1 is a future-divergence proxy; it cannot certify a structural break.',
            'Hurst uses rescaled ranges of spread increments; half-life is in trading days.',
            'Daily close-to-next-close execution; 3 bps fee + 2 bps half-spread + 5 bps slippage per fill; 10% annual borrow.',
            'Equal-dollar targets are approximate after integer rounding; dynamic Ridge beta is a feature, not the share-sizing ratio.',
            'OOB error is a bagging diagnostic, not a substitute for chronological holdout.',
            'Backtest trades are research records, separate from the five-minute paper account.'])
    Path(directory,'report.json').write_text(json.dumps(report,allow_nan=False,indent=2),encoding='utf-8')
    return report
