"""Phase 3: XLM-RoBERTa binary hate-speech training.
Run from repository root: python -m part_b.multilingual.train_xlmr
Uses Phase 1's clean CSVs; does not overwrite any existing models.
"""
from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint

BASE = Path(__file__).resolve().parent
DEFAULT_DATA = BASE / "results"
DEFAULT_OUT = DEFAULT_DATA / "xlmr_multilingual"
SPLITS = {
    "train": "train_clean.csv",
    "validation": "val_clean.csv",
    "test": "test_clean.csv",
    "mybully_external": "mybully_external_val.csv",
}


def load_frame(path: Path, limit: int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Copy the Phase 1 cleaned CSVs into --data-dir.")
    d = pd.read_csv(path)
    if not {"text", "hate_label"}.issubset(d.columns):
        raise ValueError(f"{path} requires text and hate_label columns; got {list(d.columns)}")
    d = d[d["text"].notna() & d["hate_label"].isin([0, 1])].copy()
    d["text"] = d["text"].astype(str).str.strip()
    d = d[d["text"].str.len() > 0].reset_index(drop=True)
    if limit and len(d) > limit:
        # Preserve rough class balance in quick experiments, seed makes deterministic.
        d = d.groupby("hate_label", group_keys=False).sample(
            frac=limit / len(d), random_state=42
        ).reset_index(drop=True)
        if len(d) > limit:
            d = d.iloc[:limit].copy()
    return d


def metrics_from_predictions(y: np.ndarray, preds: np.ndarray, score: np.ndarray) -> dict:
    p, r, f, support = precision_recall_fscore_support(
        y, preds, labels=[0, 1], zero_division=0
    )
    result = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, preds, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y, preds, average="weighted", zero_division=0)),
        "hate_precision": float(p[1]),
        "hate_recall": float(r[1]),
        "hate_f1": float(f[1]),
        "hate_support": int(support[1]),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None
    except ValueError:
        result["roc_auc"] = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sample", type=int, default=0, help="Max records per training split for smoke test; 0=all")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Save VRAM by recomputing activations (recommended for 4GB).")
    parser.add_argument("--optim", default="adamw_torch", choices=["adamw_torch", "adafactor"], help="adafactor uses less VRAM than AdamW.")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU not detected. Use Google Colab GPU or pass --allow-cpu (slow).")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split, filename in SPLITS.items():
        cap = args.sample if split == "train" else 0
        frames[split] = load_frame(args.data_dir / filename, cap)
        print(f"{split}: {len(frames[split]):,} samples")

    # Phase 1 already separated splits. Never concatenate them before training.
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label={0: "Non-Hate", 1: "Hate"},
        label2id={"Non-Hate": 0, "Hate": 1},
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    ds = {}
    for split, frame in frames.items():
        raw = Dataset.from_dict({
            "text": frame["text"].tolist(),
            "labels": frame["hate_label"].astype(int).tolist(),
        })
        ds[split] = raw.map(tokenize, batched=True, remove_columns=["text"])

    def trainer_metrics(result):
        logits, labels = result
        pred = np.argmax(logits, axis=1)
        return {
            "macro_f1": f1_score(labels, pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(labels, pred),
        }

    train_args = TrainingArguments(
        output_dir=str(args.out_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=bool(torch.cuda.is_available()),
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        dataloader_num_workers=0,
        logging_strategy="steps",
        logging_steps=100,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    last_checkpoint = get_last_checkpoint(
        str(args.out_dir / "checkpoints")
    )
    trainer.train(
        resume_from_checkpoint=last_checkpoint
    )
    model_dir = args.out_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    pd.DataFrame(trainer.state.log_history).to_csv(args.out_dir / "training_history.csv", index=False)

    results = []
    confusions = []
    lang_rows = []
    for split in ("train", "validation", "test", "mybully_external"):
        out = trainer.predict(ds[split])
        logits = np.asarray(out.predictions)
        score = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        pred = logits.argmax(axis=1).astype(int)
        frame = frames[split]
        y = frame["hate_label"].astype(int).to_numpy()
        results.append({"split": split, **metrics_from_predictions(y, pred, score)})
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        confusions.append({"split": split, "tn": int(tn), "fp": int(fp),
                           "fn": int(fn), "tp": int(tp)})
        if "language" in frame.columns:
            for lang, sub in frame.groupby("language", dropna=False):
                ids = sub.index.to_numpy()
                if len(ids) >= 20:
                    lang_rows.append({"split": split, "language": str(lang),
                                      **metrics_from_predictions(y[ids], pred[ids], score[ids])})
        print(f"{split}: Macro F1 = {results[-1]['macro_f1']:.4f} (n={len(y)})")

    pd.DataFrame(results).to_csv(args.out_dir / "split_metrics.csv", index=False)
    pd.DataFrame(confusions).to_csv(args.out_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame(lang_rows).to_csv(args.out_dir / "language_metrics.csv", index=False)
    with (args.out_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump({"base_model": args.model, "epochs": args.epochs,
                   "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                   "learning_rate": args.lr, "max_length": args.max_length,
                   "sample": args.sample, "seed": 42,
                   "gradient_checkpointing": args.gradient_checkpointing,
                   "optimizer": args.optim,
                   "selection": "best validation macro F1; test evaluated after training",
                   "label_meaning": {"0": "Non-Hate", "1": "Hate"}}, f, indent=2)
    print(f"Finished. Model and CSV results saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
