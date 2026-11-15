"""Actual SHAP attribution for each executed backtest entry."""
import numpy as np
import pandas as pd
from ml.report.models import probabilities


def explain_trades(bundles,trades):
    import shap
    explanations=[]
    for pair_id,bundle in bundles.items():
        indices=sorted({t['signal_index'] for t in trades if t['pair']==pair_id})
        if not indices:continue
        model=bundle['model'];columns=list(bundle['train_x'].columns)
        background=bundle['train_x'].iloc[np.linspace(0,len(bundle['train_x'])-1,min(30,len(bundle['train_x']))).astype(int)]
        rows=bundle['design'].iloc[indices]
        def score(values):
            return probabilities(model,pd.DataFrame(values,columns=columns))[:,2]
        # Model-agnostic permutation SHAP works for multiclass gradient boosting
        # as well as forests, and explains the actual selected pipeline score.
        explainer=shap.Explainer(score,background,algorithm='permutation',seed=42)
        values=explainer(rows,max_evals=3*(2*len(columns)+1))
        actual=score(rows)
        for i,index in enumerate(indices):
            base=float(values.base_values[i]);contrib=np.asarray(values.values[i],dtype=float)
            reconstructed=base+float(contrib.sum())
            if not np.isclose(reconstructed,actual[i],atol=1e-6):
                raise ValueError('SHAP additivity check failed; explanation was not published.')
            explanations.append(dict(pair=pair_id,signal_index=int(index),time=str(rows.index[i]),
                base_value=base,score=float(actual[i]),reconstructed=reconstructed,
                contributions=dict(zip(columns,map(float,contrib))),
                features={name:float(rows.iloc[i][name]) for name in columns},
                method='Permutation SHAP, three permutations, training-only background; probability output'))
    return explanations
