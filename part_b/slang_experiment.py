"""Paired validation evaluation: baseline comments vs slang-expanded comments.

This DOES NOT train a model and does not claim the normalization helps.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .common import (LABELS,load_splits,read_thresholds,metrics,score_model,
                     load_saved_model,verify_baseline_config)
from .slang import normalize_slang


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['svm','xgboost'],default='xgboost')
    p.add_argument('--csv',default='train.csv')
    p.add_argument('--sample',type=int,default=0)
    p.add_argument('--project-root',default='.')
    p.add_argument('--out',default='part_b/results')
    a=p.parse_args(); root=Path(a.project_root); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    texts,y,splits=load_splits(a.csv,a.sample)
    verify_baseline_config(root,a.csv,a.sample,len(texts))
    obj=load_saved_model(root/'results'/'models'/f'{a.model}.joblib')
    thresh=read_thresholds(root/'results'/f'{a.model}_thresholds.csv')
    ix=splits['validation']
    original=texts[ix]; normalized=[normalize_slang(x) for x in original]
    changed=np.array([a!=b for a,b in zip(original,normalized)])
    rows=[]
    # Measure ALL validation rows and separately rows modified by normalization.
    for subset,mask in [('all',np.ones(len(ix),dtype=bool)),('changed_only',changed)]:
        if not mask.any(): continue
        for variant,t in [('original',list(original[mask])),('normalized',list(np.asarray(normalized)[mask]))]:
            values=score_model(obj,a.model,t)
            rows.append({'model':a.model,'subset':subset,'variant':variant,'rows':int(mask.sum()),
                         **metrics(y[ix][mask],values,thresh)})
    path=out/f'{a.model}_slang_validation.csv';pd.DataFrame(rows).to_csv(path,index=False)
    print('Modified comments:',changed.sum(),'out of',len(ix))
    print(pd.DataFrame(rows)[['subset','variant','rows','macro_f1','micro_f1']].to_string(index=False))
    print('Saved:',path)
    print('Caution: This test may have few changed examples. Only report observed effects.')

if __name__=='__main__':main()
