"""Part B: leakage-safe SVM/XGBoost tuning and separate evaluation artifacts.

Example: python -X utf8 -m part_b.tune --model svm --csv train.csv --sample 0
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

from .common import (LABELS, SEED, load_splits, thresholds_from_validation,
                     metrics, score_model, verify_baseline_config, read_thresholds)


def candidates(model):
    if model=='svm':
        return [
            {'C':.1,'ngram_range':(1,2),'max_features':15000},
            {'C':.5,'ngram_range':(1,2),'max_features':15000},
            {'C':1.,'ngram_range':(1,2),'max_features':15000},
            {'C':.5,'ngram_range':(1,1),'max_features':20000},
        ]
    return [
        {'max_depth':3,'n_estimators':120,'learning_rate':.08},
        {'max_depth':4,'n_estimators':120,'learning_rate':.08},
        {'max_depth':4,'n_estimators':200,'learning_rate':.05},
    ]


def make_estimator(name, p):
    if name=='svm':
        return OneVsRestClassifier(LinearSVC(C=p['C'],random_state=SEED))
    from xgboost import XGBClassifier
    return OneVsRestClassifier(XGBClassifier(
        max_depth=p['max_depth'],n_estimators=p['n_estimators'],learning_rate=p['learning_rate'],
        subsample=.85,colsample_bytree=.85,tree_method='hist',n_jobs=2,
        eval_metric='logloss',random_state=SEED))


def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    texts,y,splits=load_splits(args.csv,args.sample)
    verify_baseline_config(args.project_root,args.csv,args.sample,len(texts))
    # Guard against accidentally substituting a different source CSV.
    # Dataset hash binds the text/labels & row order used for this experiment.
    import hashlib
    h=hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()
    (out/'dataset_sha256.txt').write_text(h+'  '+str(args.csv)+'\n',encoding='utf-8')
    tr,va,te=(splits['train'],splits['validation'],splits['test'])
    model_name=args.model
    search=[]; best=None
    for idx,p in enumerate(candidates(model_name),start=1):
        start=time.perf_counter()
        vectorizer=TfidfVectorizer(max_features=p.get('max_features',15000),
            ngram_range=p.get('ngram_range',(1,2)),min_df=2,sublinear_tf=True,
            strip_accents='unicode')
        xtr=vectorizer.fit_transform(texts[tr]); xva=vectorizer.transform(texts[va])
        estimator=make_estimator(model_name,p)
        estimator.fit(xtr,y[tr])
        scores=np.asarray(estimator.decision_function(xva) if model_name=='svm' else estimator.predict_proba(xva))
        thresholds=thresholds_from_validation(y[va],scores,is_svm=model_name=='svm')
        m=metrics(y[va],scores,thresholds)
        sec=time.perf_counter()-start
        row={'candidate':idx,'model':model_name,'parameters':json.dumps(p),
             'training_and_validation_seconds':round(sec,2),**m}
        search.append(row)
        print(f'Candidate {idx}: validation macro F1={m["macro_f1"]:.4f}; {sec:.1f}s')
        if best is None or m['macro_f1']>best['val']['macro_f1']:
            best={'params':p,'model':estimator,'vectorizer':vectorizer,
                  'thresholds':thresholds,'val':m}
    pd.DataFrame(search).to_csv(out/f'{model_name}_search.csv',index=False)
    joblib.dump({'model':best['model'],'vectorizer':best['vectorizer']},out/f'{model_name}_tuned.joblib')
    pd.DataFrame({'label':LABELS,'threshold':best['thresholds']}).to_csv(
        out/f'{model_name}_tuned_thresholds.csv',index=False)
    (out/f'{model_name}_best_params.json').write_text(json.dumps(best['params'],indent=2),encoding='utf-8')

    # Once after selection, evaluate final choice on holdout test; do not tune with these values.
    tuned_bundle={'model':best['model'],'vectorizer':best['vectorizer']}
    eval_rows=[]; conf_rows=[]
    for split_name,ix in splits.items():
        scores=score_model(tuned_bundle,model_name,texts[ix])
        metric=metrics(y[ix],scores,best['thresholds'])
        eval_rows.append({'model':model_name+'_tuned','split':split_name,'samples':len(ix),**metric})
        pred=(scores>=best['thresholds']).astype(int)
        for j,label in enumerate(LABELS):
            tn,fp,fn,tp=confusion_matrix(y[ix,j],pred[:,j],labels=[0,1]).ravel()
            conf_rows.append({'model':model_name+'_tuned','split':split_name,'label':label,
                              'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)})
    pd.DataFrame(eval_rows).to_csv(out/f'{model_name}_tuned_split_metrics.csv',index=False)
    pd.DataFrame(conf_rows).to_csv(out/f'{model_name}_tuned_confusion.csv',index=False)
    print(pd.DataFrame(eval_rows)[['model','split','macro_f1','micro_f1']].to_string(index=False))
    print('Saved results in:',out.resolve())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=['svm','xgboost'])
    parser.add_argument('--csv',default='train.csv')
    parser.add_argument('--sample',type=int,default=0)
    parser.add_argument('--project-root',default='.')
    parser.add_argument('--out',default='part_b/results')
    args=parser.parse_args()
    run(args)

if __name__=='__main__': main()
