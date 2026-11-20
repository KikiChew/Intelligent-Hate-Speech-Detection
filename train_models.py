"""BMDS2114 Jigsaw toxic comments: Linear SVM, XGBoost, BiLSTM, TextCNN.

Examples:
    python -X utf8 train_models.py --model all --sample 10000
    python -X utf8 train_models.py --model textcnn --sample 10000
    python -X utf8 train_models.py --model all --sample 0  # full dataset

Models use a reproducible 70/15/15 split and validation-tuned label thresholds.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 42


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)


def evaluate(y_true, values, thresholds):
    pred = (values >= thresholds).astype(int)
    result = {
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, pred, average="micro", zero_division=0)),
    }
    auc_values = []
    for i, label in enumerate(LABELS):
        result[f"{label}_f1"] = float(f1_score(y_true[:, i], pred[:, i], zero_division=0))
        if len(np.unique(y_true[:, i])) == 2:
            auc_values.append(roc_auc_score(y_true[:, i], values[:, i]))
    result["macro_roc_auc"] = float(np.mean(auc_values)) if auc_values else np.nan
    return result


def tune_thresholds(y_val, scores, svm=False):
    thresholds = []
    for i in range(len(LABELS)):
        if svm:
            # LinearSVC outputs decision margins, not probabilities.
            candidates = np.unique(np.quantile(scores[:, i], np.linspace(0.01, 0.99, 81)))
            candidates = np.unique(np.append(candidates, 0.0))
        else:
            candidates = np.arange(0.05, 0.96, 0.05)
        best = max(candidates, key=lambda t: f1_score(
            y_val[:, i], scores[:, i] >= t, zero_division=0))
        thresholds.append(float(best))
    return np.array(thresholds)


def classical(name, texts_tr, y_tr, texts_val, y_val, texts_test, models_dir):
    import joblib

    vectorizer = TfidfVectorizer(
        max_features=15000, ngram_range=(1, 2), min_df=2,
        sublinear_tf=True, strip_accents="unicode")
    x_tr = vectorizer.fit_transform(texts_tr)
    x_val = vectorizer.transform(texts_val)
    x_test = vectorizer.transform(texts_test)

    if name == "svm":
        model = OneVsRestClassifier(LinearSVC(C=1.0, random_state=SEED))
        model.fit(x_tr, y_tr)
        val_scores = model.decision_function(x_val)
        test_scores = model.decision_function(x_test)
    else:
        from xgboost import XGBClassifier
        model = OneVsRestClassifier(XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85, tree_method="hist",
            n_jobs=2, eval_metric="logloss", random_state=SEED))
        model.fit(x_tr, y_tr)
        val_scores = model.predict_proba(x_val)
        test_scores = model.predict_proba(x_test)

    thresholds = tune_thresholds(y_val, val_scores, svm=name == "svm")
    joblib.dump({"vectorizer": vectorizer, "model": model}, models_dir / f"{name}.joblib")
    return test_scores, thresholds


def neural(name, texts_tr, y_tr, texts_val, y_val, texts_test, models_dir):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.utils.set_random_seed(SEED)
    vocabulary_size = 20000
    sequence_length = 160
    vectorizer = layers.TextVectorization(
        max_tokens=vocabulary_size, output_mode="int",
        output_sequence_length=sequence_length)
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(
        tf.constant(texts_tr.tolist(), dtype=tf.string)).batch(256))

    inputs = keras.Input(shape=(), dtype=tf.string, name="comment")
    x = vectorizer(inputs)
    if name == "bilstm":
        x = layers.Embedding(vocabulary_size, 96, mask_zero=True)(x)
        x = layers.Bidirectional(layers.LSTM(48))(x)
        x = layers.Dropout(0.35)(x)
    else:  # TextCNN: convolution extracts nearby word patterns.
        x = layers.Embedding(vocabulary_size, 128)(x)
        x = layers.Conv1D(128, kernel_size=3, activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(len(LABELS), activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy")
    train_tensor = tf.constant(texts_tr.tolist(), dtype=tf.string)
    val_tensor = tf.constant(texts_val.tolist(), dtype=tf.string)
    test_tensor = tf.constant(texts_test.tolist(), dtype=tf.string)
    model.fit(
        train_tensor, y_tr.astype("float32"),
        validation_data=(val_tensor, y_val.astype("float32")),
        epochs=5, batch_size=64, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=1, restore_best_weights=True)])

    val_scores = model.predict(val_tensor, batch_size=128, verbose=0)
    test_scores = model.predict(test_tensor, batch_size=128, verbose=0)
    thresholds = tune_thresholds(y_val, val_scores)
    model.save(str(models_dir / f"{name}.keras"))
    return test_scores, thresholds



def saved_model_scores(name, texts, models_dir):
    """Predict using trained models; NO retraining or threshold retuning."""
    if name in ("svm", "xgboost"):
        import joblib
        path = models_dir / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; train this model first")
        saved = joblib.load(path)
        features = saved["vectorizer"].transform(texts)
        if name == "svm":
            return np.asarray(saved["model"].decision_function(features))
        return np.asarray(saved["model"].predict_proba(features))
    import tensorflow as tf
    from tensorflow import keras
    path = models_dir / f"{name}.keras"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; train this model first")
    model = keras.models.load_model(str(path), compile=False)
    input_tensor = tf.constant(texts.tolist(), dtype=tf.string)
    return np.asarray(model.predict(input_tensor, batch_size=128, verbose=0))


def save_diagnostics(name, texts, y, splits, thresholds, out_dir, models_dir):
    """Save per-split metrics and six per-label 2x2 confusion matrices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_rows = []
    cm_rows = []
    for split_name, idx in splits.items():
        scores = saved_model_scores(name, texts[idx], models_dir)
        actual = y[idx]
        metrics = evaluate(actual, scores, thresholds)
        metric_rows.append({"model": name, "split": split_name,
                            "samples": len(idx), **metrics})
        predicted = (scores >= thresholds).astype(int)
        fig, axs = plt.subplots(2, 3, figsize=(13, 8))
        for i, label in enumerate(LABELS):
            tn, fp, fn, tp = confusion_matrix(
                actual[:, i], predicted[:, i], labels=[0, 1]
            ).ravel()
            cm = np.array([[tn, fp], [fn, tp]])
            cm_rows.append({"model": name, "split": split_name, "label": label,
                            "TN": int(tn), "FP": int(fp),
                            "FN": int(fn), "TP": int(tp)})
            ax = axs.flat[i]
            ax.imshow(cm, cmap="Blues", vmin=0)
            for row in range(2):
                for col in range(2):
                    ax.text(col, row, str(cm[row, col]), ha="center", va="center")
            ax.set(title=label.replace("_", " "), xticks=[0, 1], yticks=[0, 1],
                   xlabel="Predicted", ylabel="Actual")
            ax.set_xticklabels(["0", "1"])
            ax.set_yticklabels(["0", "1"])
        fig.suptitle(f"{name} — {split_name} confusion matrices")
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_{split_name}_confusion_matrices.png", dpi=140)
        plt.close(fig)
    pd.DataFrame(cm_rows).to_csv(out_dir / f"{name}_confusion_matrices.csv", index=False)
    return metric_rows


def load_saved_thresholds(name, out_dir):
    path = out_dir / f"{name}_thresholds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; train this model first")
    threshold_df = pd.read_csv(path).set_index("label").reindex(LABELS)
    if threshold_df["threshold"].isna().any():
        raise ValueError(f"Invalid thresholds in {path}")
    return threshold_df["threshold"].to_numpy(dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Train Jigsaw toxicity models")
    parser.add_argument("--csv", default="train.csv")
    parser.add_argument("--model", choices=["svm", "xgboost", "bilstm", "textcnn", "all"],
                        default="all")
    parser.add_argument("--sample", type=int, default=20000,
                        help="Number of comments to sample (0 = full dataset)")
    parser.add_argument("--out", default="results")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Evaluate saved models without retraining")
    args = parser.parse_args()
    if args.sample < 0:
        parser.error("--sample must be >= 0")

    set_seed()
    out_dir = Path(args.out)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv).dropna(subset=["comment_text"]).copy()
    missing = [label for label in LABELS if label not in df.columns]
    if missing:
        raise ValueError(f"CSV lacks labels: {missing}; use Jigsaw train.csv")
    df = df[df[LABELS].isin([0, 1]).all(axis=1)].copy()
    if args.sample and args.sample < len(df):
        df, _ = train_test_split(
            df, train_size=args.sample,
            stratify=df[LABELS].max(axis=1), random_state=SEED)

    texts = df["comment_text"].astype(str).to_numpy(dtype=str)
    y = df[LABELS].to_numpy(dtype=np.int32)
    indices = np.arange(len(df))
    train_val, test = train_test_split(
        indices, test_size=0.15, stratify=y.max(axis=1), random_state=SEED)
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85,
        stratify=y[train_val].max(axis=1), random_state=SEED)
    print(f"Rows: train={len(train)}, validation={len(val)}, test={len(test)}")

    names = ["svm", "xgboost", "bilstm", "textcnn"] if args.model == "all" else [args.model]
    # A comparison is only valid when every result used the same data/split.
    # Never silently mix results from different samples or CSVs.
    signature = {"csv": str(Path(args.csv).resolve()), "sample": args.sample,
                 "rows": len(df), "seed": SEED}
    signature_file = out_dir / "experiment_config.json"
    existing = pd.DataFrame()
    if (out_dir / "model_comparison.csv").exists() and signature_file.exists():
        try:
            old_signature = json.loads(signature_file.read_text(encoding="utf-8"))
            if old_signature == signature:
                existing = pd.read_csv(out_dir / "model_comparison.csv")
            else:
                print("Different dataset/sample settings: starting a new comparison table.")
        except (ValueError, OSError):
            pass

    splits = {"train": train, "validation": val, "test": test}
    if args.evaluate_only:
        if not signature_file.exists():
            raise FileNotFoundError("experiment_config.json not found: cannot verify split")
        saved_config = json.loads(signature_file.read_text(encoding="utf-8"))
        if saved_config != signature:
            raise ValueError(
                "Dataset/sample configuration differs from saved model training. "
                "Use the same --csv, --sample, and seed to avoid misleading results."
            )
        all_metrics = []
        for name in names:
            print(f"\n=== Evaluating saved {name} ===")
            threshold_values = load_saved_thresholds(name, out_dir)
            all_metrics.extend(save_diagnostics(
                name, texts, y, splits, threshold_values, out_dir, models_dir))
        destination = out_dir / "split_metrics.csv"
        current = pd.DataFrame(all_metrics)
        if destination.exists():
            old = pd.read_csv(destination)
            current = pd.concat([old, current], ignore_index=True)
            current = current.drop_duplicates(subset=["model", "split"], keep="last")
        current.to_csv(destination, index=False)
        print("\nSaved: ", destination)
        print(current[["model", "split", "macro_f1", "micro_f1"]].to_string(index=False))
        return

    rows = []
    for name in names:
        print(f"\n=== Training {name} ===")
        if name in ("svm", "xgboost"):
            test_scores, thresholds = classical(
                name, texts[train], y[train], texts[val], y[val], texts[test], models_dir)
        else:
            test_scores, thresholds = neural(
                name, texts[train], y[train], texts[val], y[val], texts[test], models_dir)

        result = {"model": name, **evaluate(y[test], test_scores, thresholds)}
        rows.append(result)
        print(json.dumps(result, indent=2))
        pd.DataFrame({"label": LABELS, "threshold": thresholds}).to_csv(
            out_dir / f"{name}_thresholds.csv", index=False)
        pd.DataFrame(test_scores, columns=LABELS).to_csv(
            out_dir / f"{name}_test_scores.csv", index=False)
        # Save after every model, so a later error doesn't discard earlier results.
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        combined = combined.drop_duplicates(subset="model", keep="last")
        combined.to_csv(out_dir / "model_comparison.csv", index=False)
        signature_file.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    print("\nFinished. Saved models and comparison results to:", out_dir.resolve())


if __name__ == "__main__":
    main()
