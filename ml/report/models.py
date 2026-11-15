"""Stages 3/4: causal regression stacking and cost-sensitive model comparison."""
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, mean_squared_error

from ml.report.features import engineer, HORIZON

CLASSES = [-1, 0, 1]


def probabilities(model, x):
    raw=model.predict_proba(x)
    out=np.zeros((len(x),3))
    for i,label in enumerate(model.classes_): out[:,CLASSES.index(int(label))]=raw[:,i]
    return out


def weighted_loss(y, probs):
    ids=np.array([CLASSES.index(int(v)) for v in y])
    weights=np.where(np.asarray(y)==1,1.,3.)
    return float(np.mean(-weights*np.log(np.clip(probs[np.arange(len(y)),ids],1e-9,1))))


def thresholds(y, probs, z):
    result={}
    for side,mask in [('long',(z<=-2)&(z>-4)),('short',(z>=2)&(z<4))]:
        truth=np.asarray(y)[mask]==1
        scores=probs[mask,2]
        if len(truth)<5 or not truth.any(): result[side]=1.01; continue
        choices=[]
        for threshold in np.r_[np.arange(.5,1.,.05),1.01]:
            pred=scores>=threshold
            fp=int(np.sum(pred & ~truth)); fn=int(np.sum(~pred & truth))
            choices.append((3*fp+fn,-float(threshold),float(threshold)))
        result[side]=min(choices)[2]
    return result


def evaluate(y, probs, z, threshold):
    raw=np.asarray(CLASSES)[probs.argmax(axis=1)]
    truth=np.asarray(y)==1
    accepted=(np.abs(z)>=2)&(np.abs(z)<4)&(probs[:,2]>=np.where(z<0,threshold['long'],threshold['short']))
    fp=int(np.sum(accepted & ~truth)); fn=int(np.sum(~accepted & truth))
    return dict(rows=len(y),accuracy=float(accuracy_score(y,raw)),
        precision=float(precision_score(truth,accepted,zero_division=0)),
        recall=float(recall_score(truth,accepted,zero_division=0)),
        confusion=confusion_matrix(y,raw,labels=CLASSES).tolist(), false_positives=fp,false_negatives=fn,
        signals=int(accepted.sum()),cost_errors=3*fp+fn,weighted_log_loss=weighted_loss(y,probs))


def fit_classifier(estimator, x, y, weighted=True):
    weights=np.where(np.asarray(y)==1,1.,3.) if weighted else np.ones(len(y))
    if isinstance(estimator,KNeighborsClassifier):
        ids=np.repeat(np.arange(len(y)),weights.astype(int))
        return make_pipeline(MinMaxScaler(),clone(estimator)).fit(x.iloc[ids],y.iloc[ids])
    if isinstance(estimator,SVC):
        # No SVC(probability=True): its internal calibration does not respect chronology.
        cut=int(len(x)*.7)
        fit_x,fit_y=x.iloc[:cut-HORIZON-1],y.iloc[:cut-HORIZON-1]
        if set(fit_y.unique()) != set(y.unique()) or len(fit_y)<25:
            raise ValueError('Insufficient chronological SVM calibration classes.')
        base=make_pipeline(MinMaxScaler(),clone(estimator))
        base.fit(fit_x,fit_y,svc__sample_weight=weights[:cut-HORIZON-1])
        calibrated=CalibratedClassifierCV(base,method='sigmoid',cv='prefit')
        return calibrated.fit(x.iloc[cut:],y.iloc[cut:],sample_weight=weights[cut:])
    model=make_pipeline(MinMaxScaler(),clone(estimator))
    name=list(model.named_steps)[-1]
    return model.fit(x,y,**{name+'__sample_weight':weights})


def train_pair(prices,pair):
    xx,labels,target,net,spread=engineer(prices,pair)
    n=len(prices);a,b,c=int(n*.3),int(n*.6),int(n*.8)
    def ids(lo,hi):
        rows=np.arange(lo,hi-HORIZON-1)
        ok=np.isfinite(xx.iloc[rows]).all(axis=1)&labels.iloc[rows].notna()&target.iloc[rows].notna()
        return rows[ok.to_numpy()]
    it,iv,ie=ids(a,b),ids(b,c),ids(c,n)
    if min(len(it),len(iv),len(ie))<35 or labels.iloc[it].nunique()<2 or 1 not in labels.iloc[it].unique():
        raise ValueError('Insufficient causal samples or profitable training examples for the report models.')
    xt,yt=xx.iloc[it],labels.iloc[it].astype(int)
    # Choose the regressor on a chronological inner split of training only.
    regressors={'OLS':LinearRegression(),'Ridge 0.1':Ridge(alpha=.1),'Ridge 1':Ridge(alpha=1.),'Lasso 0.01':Lasso(alpha=.01,max_iter=10000)}
    cut=int(len(it)*.7); regression_rows=[];regression_models={}
    for name,estimator in regressors.items():
        inner=make_pipeline(MinMaxScaler(),clone(estimator))
        inner.fit(xt.iloc[:cut-2],target.iloc[it[:cut-2]])
        mse=float(mean_squared_error(target.iloc[it[cut:]],inner.predict(xt.iloc[cut:])))
        fitted=make_pipeline(MinMaxScaler(),clone(estimator)).fit(xt,target.iloc[it])
        regression_models[name]=fitted
        regression_rows.append(dict(model=name,inner_validation_mse=mse,
            validation_mse=float(mean_squared_error(target.iloc[iv],fitted.predict(xx.iloc[iv]))),
            test_mse=float(mean_squared_error(target.iloc[ie],fitted.predict(xx.iloc[ie])))))
    reg_name=min(regression_rows,key=lambda r:r['inner_validation_mse'])['model']
    reg=regression_models[reg_name]
    forecast=pd.Series(np.nan,index=xx.index)
    for train_ids,valid_ids in TimeSeriesSplit(n_splits=3,gap=HORIZON+1).split(xt):
        fold=clone(reg).fit(xt.iloc[train_ids],target.iloc[it[train_ids]])
        forecast.iloc[it[valid_ids]]=fold.predict(xt.iloc[valid_ids])
    future_ids=np.arange(b,n)
    valid_future=future_ids[np.isfinite(xx.iloc[future_ids]).all(axis=1)]
    forecast.iloc[valid_future]=reg.predict(xx.iloc[valid_future])
    # Lasso selection fits training only and remains frozen through validation/test.
    selector=make_pipeline(MinMaxScaler(),Lasso(alpha=.01,max_iter=10000)).fit(xt,target.iloc[it])
    coefficients=dict(zip(xx.columns,map(float,selector[-1].coef_)))
    selected=[name for name,value in coefficients.items() if abs(value)>1e-8]
    fallback=not selected
    if fallback:selected=list(xx.columns)
    design=xx[selected].copy();design['forecast_change']=forecast
    it=it[forecast.iloc[it].notna().to_numpy()]
    if len(it)<40:raise ValueError('Too few training rows after causal regression stacking.')
    train_x,train_y=design.iloc[it],labels.iloc[it].astype(int)
    valid_x,valid_y=design.iloc[iv],labels.iloc[iv].astype(int)
    test_x,test_y=design.iloc[ie],labels.iloc[ie].astype(int)
    candidates=[]
    for strength in [.1,1.]:
        candidates += [('Logistic',{'C':strength},LogisticRegression(C=strength,max_iter=2000)),
                       ('Linear SVM',{'C':strength},SVC(C=strength,kernel='linear')),
                       ('RBF SVM',{'C':strength,'gamma':'scale'},SVC(C=strength,kernel='rbf',gamma='scale'))]
    for k in [5,11]:candidates.append(('KNN',{'neighbors':k},KNeighborsClassifier(n_neighbors=k)))
    for depth in [2,4,6]:
        candidates += [('Decision tree',{'depth':depth},DecisionTreeClassifier(criterion='log_loss',max_depth=depth,min_samples_leaf=8,random_state=42)),
            ('Random forest',{'depth':depth},RandomForestClassifier(criterion='log_loss',n_estimators=120,max_depth=depth,min_samples_leaf=8,oob_score=True,random_state=42,n_jobs=1))]
    for depth in [2,3]:candidates.append(('Gradient boosting',{'depth':depth},GradientBoostingClassifier(n_estimators=80,max_depth=depth,min_samples_leaf=8,random_state=42)))
    fitted=[];errors=[]
    for name,params,estimator in candidates:
        try:
            model=fit_classifier(estimator,train_x,train_y)
            vp=probabilities(model,valid_x); th=thresholds(valid_y,vp,xx.iloc[iv].zscore.to_numpy())
            val=evaluate(valid_y,vp,xx.iloc[iv].zscore.to_numpy(),th)
            test=evaluate(test_y,probabilities(model,test_x),xx.iloc[ie].zscore.to_numpy(),th)
            final_est=model[-1] if hasattr(model,'steps') else None
            row=dict(model=name,parameters=params,thresholds=th,validation=val,test=test,
                     train_loss=weighted_loss(train_y,probabilities(model,train_x)),
                     oob_error=float(1-final_est.oob_score_) if hasattr(final_est,'oob_score_') else None)
            fitted.append((row,model,estimator))
        except ValueError as exc:errors.append(dict(model=name,parameters=params,error=str(exc)))
    ensembles=[item for item in fitted if item[0]['model'] in ('Random forest','Gradient boosting')]
    if not ensembles:raise ValueError('No ensemble model could be fitted.')
    chosen=min(ensembles,key=lambda item:(item[0]['validation']['cost_errors'],item[0]['validation']['weighted_log_loss']))
    chosen_row,chosen_model,chosen_est=chosen
    baseline=fit_classifier(chosen_est,train_x,train_y,weighted=False)
    bp=probabilities(baseline,valid_x);bt=thresholds(valid_y,bp,xx.iloc[iv].zscore.to_numpy())
    comparison=evaluate(test_y,probabilities(baseline,test_x),xx.iloc[ie].zscore.to_numpy(),bt)
    # Forecast high/low volatility, using a cutoff fitted on the training window.
    cutoff=float(xx.iloc[it].market_volatility.median())
    regime_y=(xx.market_volatility.shift(-5).iloc[it]>cutoff).astype(int)
    regime=None
    if regime_y.nunique()==2:
        regime=make_pipeline(MinMaxScaler(),LogisticRegression(max_iter=1000)).fit(xx.iloc[it],regime_y)
    signals={}
    available=np.arange(c,n);available=available[np.isfinite(design.iloc[available]).all(axis=1)]
    pp=probabilities(chosen_model,design.iloc[available])
    regimes=regime.predict(xx.iloc[available]) if regime is not None else np.zeros(len(available))
    for index,probs,regime_value in zip(available,pp,regimes):
        z=float(xx.zscore.iloc[index]); side='long' if z<0 else 'short'
        th=chosen_row['thresholds'][side]
        action='Hold'
        if int(CLASSES[int(np.argmax(probs))])==-1:action='Exit / halt'
        elif 2<=abs(z)<4 and probs[2]>=th:action=side.title()
        signals[int(index)]=dict(action=action,score=float(probs[2]),breakdown_score=float(probs[0]),
            zscore=z,beta=float(spread.beta.iloc[index]),forecast=float(forecast.iloc[index]),
            high_volatility=bool(regime_value),threshold=float(th),side=side,
            features={k:float(v) for k,v in design.iloc[index].items()})
    report=dict(pair=pair['id'],y=pair['y'],x=pair['x'],selected=deepcopy(chosen_row),
        regression=regression_rows,selected_regression=reg_name,lasso_coefficients=coefficients,
        retained_features=selected,lasso_fallback=fallback,models=[r for r,_,_ in fitted],errors=errors,
        standard_loss_test=comparison,weighted_loss_test=chosen_row['test'],
        train_rows=len(it),validation_rows=len(iv),test_rows=len(ie),train_end=str(prices.index[b-1]),
        validation_end=str(prices.index[c-1]),test_start=str(prices.index[c]),purge=HORIZON+1,
        regime_cutoff=cutoff,latest=signals.get(n-1),label_counts={str(int(k)):int(v) for k,v in train_y.value_counts().items()})
    return dict(report=report,signals=signals,model=chosen_model,design=design,train_x=train_x,
                regression=reg,spread=spread,test_start=c,baselines={r['model']:m for r,m,_ in fitted})
