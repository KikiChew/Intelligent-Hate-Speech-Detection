"""Inference helper for the Phase 3 XLM-RoBERTa binary hate model."""
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL = Path(__file__).resolve().parent / 'results' / 'xlmr_multilingual_full' / 'model'


def load_model(model_dir=DEFAULT_MODEL):
    model_dir = str(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def predict(texts, bundle, max_length=128):
    tokenizer, model = bundle
    single = isinstance(texts, str)
    texts = [texts] if single else list(texts)
    if not texts:
        return []
    device = next(model.parameters()).device
    inputs = tokenizer(texts, truncation=True, padding=True,
                       max_length=max_length, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        scores = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
    out = [{'text': t, 'label': 'Hate' if s >= 0.5 else 'Non-Hate',
            'hate_probability': round(float(s), 4)} for t, s in zip(texts, scores)]
    return out[0] if single else out

if __name__ == '__main__':
    bundle = load_model()
    print(predict(['Terima kasih bro!', 'Kau ni bodoh lah.'], bundle))
