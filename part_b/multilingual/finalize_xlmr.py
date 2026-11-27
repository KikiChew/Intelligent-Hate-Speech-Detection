"""Export and evaluate an existing full-data XLM-R checkpoint WITHOUT training.

Run from repository root:
 python -X utf8 -m part_b.multilingual.finalize_xlmr

Default: checkpoint-3118 (epoch 2; best reported validation Macro F1),
output: results/xlmr_multilingual_full/ (separate from old sample results).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parent
SPLITS = {
    'train': 'train_clean.csv',
    'validation': 'val_clean.csv',
    'test': 'test_clean.csv',
    'mybully_external': 'mybully_external_val.csv',
}

def summarize(y, pred, prob):
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        'n': int(len(y)), 'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y, pred, average='micro', zero_division=0)),
        'weighted_f1': float(f1_score(y, pred, average='weighted', zero_division=0)),
        'hate_precision': float(precision[1]), 'hate_recall': float(recall[1]),
        'hate_f1': float(f1[1]), 'hate_support': int(support[1]),
        'roc_auc': float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
    }

def main():
    ap = argparse.ArgumentParser(description='Finalize existing XLM-R checkpoint (no training)')
    ap.add_argument('--data-dir', type=Path, default=ROOT / 'results')
    ap.add_argument('--source-dir', type=Path, default=ROOT / 'results' / 'xlmr_multilingual')
    ap.add_argument('--checkpoint', default='checkpoint-3118')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'results' / 'xlmr_multilingual_full')
    ap.add_argument('--max-length', type=int, default=64)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--skip-train-eval', action='store_true', help='Skip slower train-set metrics; still evaluate val/test/external')
    args = ap.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        checkpoint = args.source_dir / 'checkpoints' / args.checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f'Checkpoint missing: {checkpoint}')
    if not ((checkpoint / 'model.safetensors').exists() or (checkpoint / 'pytorch_model.bin').exists()):
        raise FileNotFoundError(f'No model weights in {checkpoint}. See checkpoint contents.')
    if args.out_dir.resolve() == args.source_dir.resolve():
        raise ValueError('Choose a separate --out-dir so old files are not overwritten before validation.')
    print('Loading FULL-DATA checkpoint:', checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint), local_files_only=True)
    # Original training used FacebookAI/xlm-roberta-base tokenizer. Prefer the tokenizer saved with the
    # older run, if available, to avoid requiring network access. It is the same tokenizer family.
    tok_dir = args.source_dir / 'model'
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir) if (tok_dir / 'tokenizer_config.json').exists() else 'FacebookAI/xlm-roberta-base', use_fast=True)
    # Save first to a NEW directory. Existing sample model remains untouched.
    out_model = args.out_dir / 'model'
    out_model.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_model), safe_serialization=True)
    tokenizer.save_pretrained(str(out_model))
    print('Saved model to:', out_model)
    ds = {}
    frames = {}
    for split, name in SPLITS.items():
        if split == 'train' and args.skip_train_eval:
            continue
        path = args.data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing cleaned CSV: {path}')
        frame = pd.read_csv(path)
        if not {'text', 'hate_label'}.issubset(frame.columns):
            raise ValueError(f'{path} missing text/hate_label columns')
        frame = frame.loc[frame['text'].notna() & frame['hate_label'].isin([0, 1])].copy()
        frame['text'] = frame['text'].astype(str).str.strip()
        frame = frame.loc[frame['text'].str.len() > 0].reset_index(drop=True)
        frames[split] = frame
        raw = Dataset.from_dict({'text': frame['text'].tolist(), 'labels': frame['hate_label'].astype(int).tolist()})
        ds[split] = raw.map(lambda batch: tokenizer(batch['text'], truncation=True, max_length=args.max_length), batched=True, remove_columns=['text'])
        print(split, 'rows:', len(frame))
    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir=str(args.out_dir / '_eval_tmp'), per_device_eval_batch_size=args.batch_size,
        report_to='none', dataloader_num_workers=0, fp16=bool(torch.cuda.is_available())),
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer))
    metrics_rows, confusion_rows, language_rows = [], [], []
    for split in SPLITS:
        if split not in ds:
            continue
        result = trainer.predict(ds[split])
        logits = np.asarray(result.predictions)
        pred = logits.argmax(axis=1).astype(int)
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()[:, 1]
        frame = frames[split]
        true = frame['hate_label'].astype(int).to_numpy()
        metrics_rows.append({'split': split, **summarize(true, pred, probs)})
        tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
        confusion_rows.append({'split': split, 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
        if 'language' in frame.columns:
            for lang, indices in frame.groupby('language', dropna=False).indices.items():
                if len(indices) >= 20:
                    ix = np.asarray(indices)
                    language_rows.append({'split': split, 'language': str(lang), **summarize(true[ix], pred[ix], probs[ix])})
        print(f"{split}: macro F1={metrics_rows[-1]['macro_f1']:.4f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(args.out_dir / 'split_metrics.csv', index=False)
    pd.DataFrame(confusion_rows).to_csv(args.out_dir / 'confusion_matrix.csv', index=False)
    pd.DataFrame(language_rows).to_csv(args.out_dir / 'language_metrics.csv', index=False)
    (args.out_dir / 'finalization_config.json').write_text(json.dumps({
        'checkpoint': str(checkpoint), 'max_length': args.max_length,
        'batch_size': args.batch_size, 'train_set_evaluated': not args.skip_train_eval,
        'note': 'Evaluation/export only; no training performed.'
    }, indent=2), encoding='utf-8')
    print('Finished! Results saved in:', args.out_dir)
    print('Original sample files left unchanged in:', args.source_dir)

if __name__ == '__main__':
    main()
