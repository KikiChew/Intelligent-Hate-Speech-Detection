"""Reproduce the existing Part A split and model conventions exactly."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
SEED = 42


def load_splits(csv_path='train.csv', sample=0):
    df = pd.read_csv(csv_path).dropna(subset=['comment_text']).copy()
    missing = [c for c in LABELS if c not in df]
    if missing:
        raise ValueError(f'Missing labels: {missing}')
    df = df[df[LABELS].isin([0, 1]).all(axis=1)].copy()
    if sample and sample < len(df):
        df, _ = train_test_split(df, train_size=sample,
                                 stratify=df[LABELS].max(axis=1),random_state=SEED)
    texts = df['comment_text'].astype(str).to_numpy(dtype=str)
    labels = df[LABELS].to_numpy(dtype=np.int32)
    ix = np.arange(len(df))
    train_val, test = train_test_split(ix, test_size=.15,
                                     stratify=labels.max(axis=1),random_state=SEED)
    train, val = train_test_split(train_val, test_size=.15/.85,
                                stratify=labels[train_val].max(axis=1),random_state=SEED)
    return texts, labels, {'train':train,'validation':val,'test':test}


def thresholds_from_validation(y, values, is_svm):
    """Same tuning rule as Part A; use validation labels only."""
    from sklearn.metrics import f1_score
    choices=[]
    for j in range(len(LABELS)):
        if is_svm:
            candidates = np.unique(np.append(np.quantile(values[:,j], np.linspace(.01,.99,81)),0.))
        else:
            candidates = np.arange(.05,.96,.05)
        t = max(candidates,key=lambda x:f1_score(y[:,j],values[:,j]>=x,zero_division=0))
        choices.append(float(t))
    return np.asarray(choices)


def metrics(y, scores, thresholds):
    from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score
    pred=(scores>=thresholds).astype(int)
    result = {'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
              'micro_f1':float(f1_score(y,pred,average='micro',zero_division=0)),
              'micro_precision':float(precision_score(y,pred,average='micro',zero_division=0)),
              'micro_recall':float(recall_score(y,pred,average='micro',zero_division=0))}
    result['macro_roc_auc']=float(np.mean([roc_auc_score(y[:,j],scores[:,j]) for j in range(y.shape[1]) if len(np.unique(y[:,j]))==2]))
    for j, label in enumerate(LABELS):
        result[label+'_f1']=float(f1_score(y[:,j],pred[:,j],zero_division=0))
    return result


def load_saved_model(path):
    import joblib
    return joblib.load(path)


def score_model(obj, name, texts):
    X=obj['vectorizer'].transform(list(texts))
    estimator=obj['model']
    values=(estimator.decision_function(X) if name=='svm' else estimator.predict_proba(X))
    return np.asarray(values)


def read_thresholds(path):
    tab=pd.read_csv(path).set_index('label').reindex(LABELS)
    if tab.threshold.isna().any():
        raise ValueError(f'Threshold labels incomplete in {path}')
    return tab.threshold.to_numpy(float)


def verify_baseline_config(project_root, csv_path, sample, nrows):
    """The baseline file has a nonportable Windows path; compare stable fields instead."""
    import json
    cfg_path=Path(project_root)/'results'/'experiment_config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f'Baseline configuration missing: {cfg_path}')
    cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
    for k, expected in {'sample':sample,'rows':nrows,'seed':SEED}.items():
        if cfg.get(k)!=expected:
            raise ValueError(f'Part A mismatch: {k} baseline={cfg.get(k)} current={expected}')
    # Still require dataset identity to be verified separately, since row count alone is insufficient.
    return cfg
