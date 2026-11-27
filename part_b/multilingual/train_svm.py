"""Train and evaluate an English–Malay binary hate-speech SVM baseline.

Run from repository root:
    python -X utf8 -m part_b.multilingual.train_svm

Reads existing Phase 1 cleaned CSV files. Never changes Phase 1 or Part A/B models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / 'results'
DEFAULT_OUT = ROOT / 'results' / 'svm_multilingual'
SPLITS = {'train': 'train_clean.csv', 'validation': 'val_clean.csv',
          'test': 'test_clean.csv', 'mybully_external': 'mybully_external_val.csv'}


def read_splits(folder: Path):
    out = {}
    for split, name in SPLITS.items():
        path = folder / name
        if not path.exists():
            raise FileNotFoundError(f'{path} is missing. First run Phase 1 prepare_data.py.')
        df = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False)
        need = {'text', 'hate_label', 'language', 'dataset'}
        if missing := (need - set(df.columns)):
            raise ValueError(f'{path} missing columns: {sorted(missing)}')
        df['hate_label'] = pd.to_numeric(df['hate_label'], errors='raise').astype(int)
        if df['text'].astype(str).str.strip().eq('').any() or not df['hate_label'].isin([0, 1]).all():
            raise ValueError(f'Bad text or binary labels in {path}')
        out[split] = df
    return out


def dup_key(v):
    import re, unicodedata
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(v)).casefold()).strip()


def assert_no_leakage(dfs):
    seen = {}
    for split, df in dfs.items():
        vals = set(df['text'].map(dup_key))
        for prior, prior_vals in seen.items():
            overlap = vals & prior_vals
            if overlap:
                raise ValueError(f'Overlap between {split} and {prior}: {len(overlap)} normalized texts')
        seen[split] = vals


def metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    vals = dict(n=len(y), hate_support=int(np.sum(y)), threshold=float(threshold),
                accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average='macro', zero_division=0),
                binary_f1=f1_score(y, pred, average='binary', zero_division=0),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                tn=tn, fp=fp, fn=fn, tp=tp)
    vals['roc_auc'] = (float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float('nan'))
    return vals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--c-values', nargs='+', type=float, default=[0.25, 1.0, 2.0])
    parser.add_argument('--word-features', type=int, default=20000)
    parser.add_argument('--char-features', type=int, default=30000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dfs = read_splits(args.data_dir)
    assert_no_leakage(dfs)
    train = dfs['train']; val = dfs['validation']
    word = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2,
                           max_features=args.word_features, strip_accents='unicode')
    char = TfidfVectorizer(analyzer='char', ngram_range=(2,5), sublinear_tf=True,
                           min_df=2, max_features=args.char_features, strip_accents='unicode')
    print('Fitting word and character TF-IDF on TRAIN ONLY...', flush=True)
    x_train = hstack([word.fit_transform(train.text.astype(str)), char.fit_transform(train.text.astype(str))], format='csr')
    x_val = hstack([word.transform(val.text.astype(str)), char.transform(val.text.astype(str))], format='csr')
    y_train = train.hate_label.to_numpy(); y_val = val.hate_label.to_numpy()
    print('Feature matrix:', x_train.shape, flush=True)
    # Decision-score thresholds are selected on VALIDATION only, not test.
    candidates = []
    best = None
    for c in args.c_values:
        model = LinearSVC(C=c, class_weight='balanced', max_iter=5000, random_state=42)
        model.fit(x_train, y_train)
        scores = model.decision_function(x_val)
        for threshold in np.linspace(-0.4, 0.4, 17):
            row = metrics(y_val, scores, float(threshold))
            candidates.append({'C': c, **row})
            rank = (row['macro_f1'], -abs(threshold), -c)
            if best is None or rank > best[0]:
                best = (rank, model, c, float(threshold))
        print(f'C={c} complete', flush=True)
    pd.DataFrame(candidates).sort_values('macro_f1', ascending=False).to_csv(args.output_dir/'svm_search.csv', index=False)
    _, model, best_c, threshold = best
    print(f'Best from validation: C={best_c}; decision threshold={threshold}', flush=True)
    # No model refit with validation: preserves a clean held-out validation selection protocol.
    model_package = {'word_vectorizer': word, 'char_vectorizer': char, 'classifier': model,
                     'threshold': threshold, 'label_map': {'0': 'Non-Hate', '1': 'Hate'},
                     'languages': 'English, Malay; code-mixed performance not separately validated'}
    joblib.dump(model_package, args.output_dir/'multilingual_svm.joblib', compress=3)
    all_metrics, confusions, slices = [], [], []
    for split, df in dfs.items():
        x = x_train if split == 'train' else x_val if split == 'validation' else hstack([
            word.transform(df.text.astype(str)), char.transform(df.text.astype(str))], format='csr')
        y = df.hate_label.to_numpy(); scores = model.decision_function(x)
        m = metrics(y, scores, threshold)
        all_metrics.append({'split': split, **m})
        confusions.append({'split': split, **{k: m[k] for k in ['tn','fp','fn','tp']}})
        for lang, idx in df.groupby('language', dropna=False).indices.items():
            if len(idx) < 2: continue
            lm = metrics(y[idx], scores[idx], threshold)
            slices.append({'split': split, 'language': lang, **lm})
        print(split, 'n=', len(df), 'Macro F1=', round(m['macro_f1'], 4),
              'Hate F1=', round(m['binary_f1'],4), flush=True)
    pd.DataFrame(all_metrics).to_csv(args.output_dir/'svm_split_metrics.csv', index=False)
    pd.DataFrame(confusions).to_csv(args.output_dir/'svm_confusion.csv', index=False)
    pd.DataFrame(slices).to_csv(args.output_dir/'svm_language_metrics.csv', index=False)
    config = {'model': 'TF-IDF word+character n-gram LinearSVC', 'best_C': best_c,
              'threshold_type': 'decision_function (NOT probability)', 'best_validation_threshold': threshold,
              'input_data_dir': str(args.data_dir), 'n_train': len(train),
              'validation_selection_metric': 'macro_f1', 'word_vocabulary': len(word.vocabulary_),
              'char_vocabulary': len(char.vocabulary_), 'evaluated_splits': list(SPLITS),
              'selection_note': 'C and threshold picked on bilingual validation only; test/external are evaluation-only.',
              'created_utc': datetime.now(timezone.utc).isoformat()}
    (args.output_dir/'svm_best_params.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Saved all files to:', args.output_dir.resolve(), flush=True)


if __name__ == '__main__':
    main()
