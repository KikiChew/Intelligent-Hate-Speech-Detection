"""Preparation for Malay-English hate speech and Manglish analysis.

Run from repository root:
    python -X utf8 -m part_b.multilingual.prepare_data

Prerequisites: pandas, scikit-learn. Run the separate dataset_analysis module
for charts after this script finishes.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent

def key_for(text: object) -> str:
    """Conservative normalization used for exact-ish duplicate grouping only.

    Keep punctuation and Unicode script distinctions instead of aggressively
    transliterating colloquial language (e.g. Malay rojak / Chinese).
    """
    s = unicodedata.normalize('NFKC', str(text)).casefold()
    return re.sub(r'\s+', ' ', s).strip()

def read_file(path: Path, expected: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing dataset: {path}\nSee README_PHASE1.md for the correct folder.')
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f'{path.name}: Missing required columns {sorted(missing)}; found {list(df.columns)}')
    return df

def build_mybully(df: pd.DataFrame, original_split: str) -> pd.DataFrame:
    # MYBully Hate is Yes/No; Bully is a separate task, not merged with Hate.
    d = df.copy()
    d['hate_label'] = d['Hate'].astype('string').str.strip().str.casefold().map({'yes':1, 'no':0})
    out = pd.DataFrame({
        'text': d['Tweet'].astype('string').str.strip(),
        'hate_label': d['hate_label'],
        'dataset': 'MYBully',
        'source_detail': 'MYBully',
        'language': 'ms_or_mixed_unverified',
        'annotation_type': d['anno_type'].fillna('unknown').astype(str),
        'bully_original': d['Bully'].fillna('').astype(str),
        'original_split': original_split,
    })
    return out

def build_bilingual(df: pd.DataFrame) -> pd.DataFrame:
    labels = pd.to_numeric(df['label'], errors='coerce')
    return pd.DataFrame({
        'text': df['text'].astype('string').str.strip(),
        'hate_label': labels,
        'dataset': 'Bilingual_MS_EN',
        'source_detail': df['source'].fillna('unknown').astype(str),
        'language': df['lang'].fillna('unknown').astype(str),
        'annotation_type': 'mixed_quality_see_source',
        'bully_original': '',
        'original_split': 'unassigned',
    })

def prune_and_audit(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    raw_n = len(df)
    d = df.copy()
    d['text'] = d['text'].astype('string').str.strip()
    valid = d['text'].notna() & d['text'].str.len().gt(0) & d['hate_label'].isin([0,1])
    removed_invalid = int((~valid).sum())
    d = d.loc[valid].copy()
    d['hate_label'] = d['hate_label'].astype(int)
    d['_key'] = d['text'].map(key_for)
    # Remove cases where *same* text has incompatible labels; no arbitrary winner.
    conflicted = set(d.groupby('_key')['hate_label'].nunique().loc[lambda x: x > 1].index)
    conflicting_rows = int(d['_key'].isin(conflicted).sum())
    d = d.loc[~d['_key'].isin(conflicted)].copy()
    before_dup = len(d)
    # Keep separate provenance in summary; a single normalized text should not
    # be counted more than once within a dataset.
    d = d.drop_duplicates('_key', keep='first').reset_index(drop=True)
    return d, {'dataset':name, 'input_rows':raw_n, 'invalid_or_missing_rows':removed_invalid,
               'ambiguous_label_rows':conflicting_rows, 'duplicate_rows_dropped':before_dup-len(d),
               'usable_unique_rows':len(d)}

def grouped_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int):
    """Split one row per normalized unique text, stratifying if possible."""
    if len(df) < 50:
        raise ValueError('Too few bilingual rows to split reliably.')
    try:
        train, hold = train_test_split(df, test_size=val_frac+test_frac,
                                       stratify=df['hate_label'], random_state=seed)
        val, test = train_test_split(hold, test_size=test_frac/(val_frac+test_frac),
                                    stratify=hold['hate_label'], random_state=seed)
    except ValueError:
        train, hold = train_test_split(df, test_size=val_frac+test_frac, random_state=seed)
        val, test = train_test_split(hold, test_size=test_frac/(val_frac+test_frac), random_state=seed)
    return train.copy(), val.copy(), test.copy()

def assert_no_leakage(groups: dict[str, pd.DataFrame]):
    for a, adf in groups.items():
        for b, bdf in groups.items():
            if a >= b:
                continue
            overlap = set(adf['_key']).intersection(set(bdf['_key']))
            if overlap:
                raise AssertionError(f'Leakage between {a} and {b}: {len(overlap)} duplicate keys')

def save(df: pd.DataFrame, path: Path):
    out = df.drop(columns=['_key'], errors='ignore').copy()
    out.to_csv(path, index=False, encoding='utf-8-sig')

def run(data_dir: Path, output_dir: Path, seed: int = 42):
    output_dir.mkdir(parents=True, exist_ok=True)
    m_train_raw = read_file(data_dir/'mybully/train.csv', {'Tweet','Hate','Bully','anno_type'})
    m_val_raw = read_file(data_dir/'mybully/val.csv', {'Tweet','Hate','Bully','anno_type'})
    b_raw = read_file(data_dir/'bilingual/bilingual_hatespeech_ms_en.csv', {'text','label','source','lang'})
    manglish_train = read_file(data_dir/'manglish/train.csv', {'text','sentiment','language','is_code_switch'})
    manglish_test = read_file(data_dir/'manglish/test.csv', {'text','sentiment','language','is_code_switch'})

    mt, a = prune_and_audit(build_mybully(m_train_raw,'original_train'), 'MYBully_train')
    mv, b = prune_and_audit(build_mybully(m_val_raw,'original_val'), 'MYBully_val')
    bi, c = prune_and_audit(build_bilingual(b_raw),'Bilingual_MS_EN')
    # PRIORITY: pre-existing MYBully validation, no training leakage.
    # Exclude any overlap with held-out MYBully val, from both training sources.
    heldout = set(mv['_key'])
    overlap_my_train = int(mt['_key'].isin(heldout).sum())
    overlap_bi = int(bi['_key'].isin(heldout).sum())
    mt = mt.loc[~mt['_key'].isin(heldout)].copy()
    bi = bi.loc[~bi['_key'].isin(heldout)].copy()

    # Also avoid overlaps across training datasets; prioritize MYBully.
    bilingual_overlap_train = int(bi['_key'].isin(set(mt['_key'])).sum())
    bi = bi.loc[~bi['_key'].isin(set(mt['_key']))].copy()

    bi_train, bi_val, bi_test = grouped_split(bi, .10, .10, seed)
    train = pd.concat([mt,bi_train], ignore_index=True)
    val = bi_val.reset_index(drop=True)
    test = bi_test.reset_index(drop=True)
    ext_val = mv.reset_index(drop=True)
    splits = {'train':train, 'validation':val, 'test':test, 'mybully_external_validation':ext_val}
    assert_no_leakage(splits)
    for filename, frame in [('train_clean.csv',train),('val_clean.csv',val),('test_clean.csv',test),
                            ('mybully_external_val.csv',ext_val)]:
        save(frame,output_dir/filename)

    quality = pd.DataFrame([a,b,c])
    quality.to_csv(output_dir/'data_quality_audit.csv',index=False)
    overlap_audit = pd.DataFrame([
       {'check':'MYBully original train vs original val','removed_from':'MYBully train','n_rows':overlap_my_train},
       {'check':'Bilingual vs MYBully original val','removed_from':'Bilingual','n_rows':overlap_bi},
       {'check':'Bilingual vs retained MYBully train','removed_from':'Bilingual','n_rows':bilingual_overlap_train},
    ])
    overlap_audit.to_csv(output_dir/'overlap_audit.csv',index=False)
    summary = pd.DataFrame([{'split':name,'dataset':ds,'hate_label':int(label),'rows':int(n)}
         for name,fr in splits.items()
         for (ds,label),n in fr.groupby(['dataset','hate_label']).size().items()])
    summary.to_csv(output_dir/'dataset_summary.csv',index=False)
    source_dist = pd.DataFrame([
        {'dataset': name, 'source_detail':source, 'language':lang, 'hate_label':int(label), 'rows':int(n)}
        for name,fr in splits.items()
        for (source,lang,label),n in fr.groupby(['source_detail','language','hate_label']).size().items()
    ])
    source_dist.to_csv(output_dir/'source_language_label_distribution.csv',index=False)

    m_summaries=[]
    for split, df in [('train',manglish_train),('test',manglish_test)]:
        for (lang,cs),count in df.groupby(['language','is_code_switch'],dropna=False).size().items():
            m_summaries.append({'split':split,'language':str(lang),'is_code_switch':str(cs),'rows':int(count)})
    pd.DataFrame(m_summaries).to_csv(output_dir/'manglish_profile.csv',index=False)
    manglish_quality = pd.DataFrame([{'split':s,'rows':len(fr),'missing_text':int(fr['text'].isna().sum()),
          'unique_text':int(fr['text'].dropna().map(key_for).nunique()),
          'augmented_rows':int(fr['is_augmented'].eq(True).sum()) if 'is_augmented' in fr else None}
          for s,fr in [('train',manglish_train),('test',manglish_test)]])
    manglish_quality.to_csv(output_dir/'manglish_quality.csv',index=False)

    info={'random_state':seed,'splits':{k:len(v) for k,v in splits.items()},
          'label_semantics_warning':'MYBully Hate and Bilingual binary hate may not have equivalent annotation guidelines. Treat merged model as exploratory.',
          'bilingual_language_counts':{str(k):int(v) for k,v in b_raw['lang'].value_counts(dropna=False).items()},
          'bilingual_source_counts':{str(k):int(v) for k,v in b_raw['source'].value_counts(dropna=False).items()},
          'mybully_val_policy':'Original MYBully validation kept separate as external evaluation; any overlaps removed from train/bilingual.',
          'manglish_policy':'No hate labels, used for EDA only; sentiment is NOT hate.',
          'dedup_policy':'NFKC + casefold + whitespace; exact-ish matches only, not semantic near-duplicates.',
          'test_policy':'The bilingual held-out test is created from original source data. No tuning on test.'}
    (output_dir/'preparation_metadata.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nPrepared splits:',info['splits'])
    print('Saved to:',output_dir.resolve())
    return info

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'datasets')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--seed',type=int,default=42)
    a=parser.parse_args()
    run(a.data_dir,a.output_dir,a.seed)

if __name__=='__main__':
    main()
