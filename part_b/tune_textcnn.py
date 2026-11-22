"""Part B TextCNN tuning; same Jigsaw split as Part A, separate artifacts.
Run in repository root: python -X utf8 -m part_b.tune_textcnn --sample 0
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .common import LABELS, SEED, load_splits, metrics, thresholds_from_validation, verify_baseline_config

# Deliberately small, documented manual search: reasonable for CPU-based coursework.
CANDIDATES = [
    {"filters": 96, "kernel_size": 3, "dropout": 0.40, "learning_rate": 0.001},
    {"filters": 128, "kernel_size": 3, "dropout": 0.50, "learning_rate": 0.0005},
    {"filters": 128, "kernel_size": 5, "dropout": 0.40, "learning_rate": 0.0005},
]


def build_model(train_text, settings, vocabulary=20000, seq_len=160):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.utils.set_random_seed(SEED)
    vectorizer = layers.TextVectorization(
        max_tokens=vocabulary, output_mode="int", output_sequence_length=seq_len
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_text).batch(256))
    inp = keras.Input(shape=(), dtype=tf.string, name="comment")
    x = vectorizer(inp)
    x = layers.Embedding(vocabulary, 128)(x)
    x = layers.Conv1D(settings["filters"], settings["kernel_size"], activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(settings["dropout"])(x)
    out = layers.Dense(len(LABELS), activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=settings["learning_rate"]),
                  loss="binary_crossentropy")
    return model


def predict(model, strings, batch_size=128):
    import tensorflow as tf
    batches = []
    for offset in range(0, len(strings), 2048):
        batch = tf.constant(strings[offset:offset+2048].tolist(), dtype=tf.string)
        batches.append(model.predict(batch, batch_size=batch_size, verbose=0))
    return np.concatenate(batches, axis=0)


def main():
    p = argparse.ArgumentParser(description="Tune TextCNN using a fixed Part A split")
    p.add_argument('--csv', default='train.csv')
    p.add_argument('--sample', type=int, default=0, help='0 = full original dataset')
    p.add_argument('--out', default='part_b/results')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    if args.sample < 0 or args.epochs < 1 or args.batch_size < 1:
        p.error('sample must be >= 0; epochs and batch-size must be positive')

    import tensorflow as tf
    from tensorflow import keras
    random.seed(SEED)
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    texts, y, splits = load_splits(args.csv, args.sample)
    verify_baseline_config('.', args.csv, args.sample, len(texts))
    tr, va = splits['train'], splits['validation']
    # tf.string avoids NumPy Unicode dtype errors on Windows.
    xtr = tf.constant(texts[tr].tolist(), dtype=tf.string)
    xva = tf.constant(texts[va].tolist(), dtype=tf.string)
    search_rows = []
    best = None
    best_path = out / 'textcnn_tuned.keras'
    for number, cfg in enumerate(CANDIDATES, 1):
        print(f'\nTextCNN candidate {number}/{len(CANDIDATES)}: {cfg}')
        keras.backend.clear_session()
        start = time.perf_counter()
        model = build_model(xtr, cfg)
        history = model.fit(xtr, y[tr].astype('float32'),
                            validation_data=(xva, y[va].astype('float32')),
                            epochs=args.epochs, batch_size=args.batch_size, verbose=2,
                            callbacks=[keras.callbacks.EarlyStopping(
                                monitor='val_loss', patience=1, restore_best_weights=True)])
        scores = predict(model, texts[va])
        thresholds = thresholds_from_validation(y[va], scores, is_svm=False)
        result = metrics(y[va], scores, thresholds)
        item = {'candidate':number, 'parameters':json.dumps(cfg),
                'epochs_ran':len(history.history['loss']),
                'training_and_validation_seconds':round(time.perf_counter()-start, 1), **result}
        search_rows.append(item)
        pd.DataFrame(search_rows).to_csv(out / 'textcnn_search.csv', index=False)
        print('Validation Macro F1:', round(result['macro_f1'], 5))
        if best is None or result['macro_f1'] > best['macro_f1']:
            model.save(str(best_path))
            pd.DataFrame({'label':LABELS, 'threshold':thresholds}).to_csv(
                out / 'textcnn_tuned_thresholds.csv', index=False)
            (out / 'textcnn_best_params.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
            pd.DataFrame(history.history).to_csv(out / 'textcnn_tuned_training_history.csv', index=False)
            best = {'macro_f1': result['macro_f1'], 'config': cfg}
        del model

    # Load only the selected model; test is NOT used for selecting candidate.
    keras.backend.clear_session()
    model = keras.models.load_model(str(best_path), compile=False)
    threshold_file = pd.read_csv(out / 'textcnn_tuned_thresholds.csv')
    thresholds = threshold_file.set_index('label').reindex(LABELS)['threshold'].to_numpy(float)
    metric_rows, confusion_rows = [], []
    for split, ids in splits.items():
        scores = predict(model, texts[ids])
        metric_rows.append({'model':'textcnn_tuned', 'split':split, 'samples':len(ids),
                            **metrics(y[ids], scores, thresholds)})
        flags = scores >= thresholds[None, :]
        for j, label in enumerate(LABELS):
            tn, fp, fn, tp = confusion_matrix(y[ids,j], flags[:,j], labels=[0,1]).ravel()
            confusion_rows.append({'model':'textcnn_tuned','split':split,'label':label,
                                   'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)})
    pd.DataFrame(metric_rows).to_csv(out / 'textcnn_tuned_split_metrics.csv', index=False)
    pd.DataFrame(confusion_rows).to_csv(out / 'textcnn_tuned_confusion.csv', index=False)
    print(pd.DataFrame(metric_rows)[['split','macro_f1','micro_f1']].to_string(index=False))
    print('Saved:', out.resolve())

if __name__ == '__main__':
    main()
