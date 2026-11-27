"""Inference for the Phase 2 binary hate speech SVM (not the six-label Jigsaw model)."""
from pathlib import Path
import joblib
from scipy.sparse import hstack

DEFAULT_MODEL = Path(__file__).resolve().parent / 'results' / 'svm_multilingual' / 'multilingual_svm.joblib'


def load_model(path=DEFAULT_MODEL):
    return joblib.load(path)  # Only load files you trust.


def predict(texts, package):
    if isinstance(texts, str):
        texts = [texts]
    x = hstack([package['word_vectorizer'].transform(texts),
                package['char_vectorizer'].transform(texts)], format='csr')
    scores = package['classifier'].decision_function(x)
    return [{'text': text, 'label': 'Hate' if s >= package['threshold'] else 'Non-Hate',
             'decision_score': float(s)} for text, s in zip(texts, scores)]


if __name__ == '__main__':
    pkg = load_model()
    for row in predict(['Terima kasih bro, sangat membantu!', 'You are an idiot.'], pkg):
        print(row)
