"""Streamlit demo for saved Jigsaw toxic-comment models.

Train at least one model first, e.g.:
    python train_models.py --model svm --sample 10000
Then run:
    streamlit run app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
DISPLAY = ['Toxic', 'Severe toxic', 'Obscene', 'Threat', 'Insult', 'Identity hate']
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODELS = RESULTS / 'models'

st.set_page_config(page_title='Toxic Comment Classifier', page_icon='🛡️', layout='wide')
st.title('🛡️ Toxic Comment Classification')
st.write('Multi-label NLP demo · BMDS2114 Machine Learning · 4-model comparison')
st.caption('Research prototype only. Predictions can be mistaken or biased; do not use as the sole basis for moderation decisions.')


def available_models():
    candidates = {
        'Linear SVM': ('svm', MODELS / 'svm.joblib'),
        'XGBoost': ('xgboost', MODELS / 'xgboost.joblib'),
        'BiLSTM': ('bilstm', MODELS / 'bilstm.keras'),
        'TextCNN': ('textcnn', MODELS / 'textcnn.keras'),
    }
    return {label: key for label, (key, path) in candidates.items() if path.exists()}


@st.cache_resource(show_spinner='Loading trained model…')
def load_model(key):
    if key in ('svm', 'xgboost'):
        import joblib
        return joblib.load(MODELS / f'{key}.joblib')
    if key in ('bilstm', 'textcnn'):
        from tensorflow import keras
        return keras.models.load_model(MODELS / f'{key}.keras', compile=False)
    raise ValueError('Unknown model')


def predict(key, obj, comment):
    if key in ('svm', 'xgboost'):
        X = obj['vectorizer'].transform([comment])
        raw = obj['model'].decision_function(X) if key == 'svm' else obj['model'].predict_proba(X)
        return np.asarray(raw)[0].astype(float)
    if key in ('bilstm', 'textcnn'):
        import tensorflow as tf
        text_input = tf.constant([comment], dtype=tf.string)
        return np.asarray(obj.predict(text_input, verbose=0))[0].astype(float)
    raise ValueError('Unknown model')


def thresholds_for(key):
    path = RESULTS / f'{key}_thresholds.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing tuned thresholds: {path.name}. Retrain this model.')
    df = pd.read_csv(path).set_index('label').reindex(LABELS)
    if df['threshold'].isna().any():
        raise ValueError('Threshold file is missing one or more labels.')
    return df['threshold'].to_numpy(dtype=float)


available = available_models()
if not available:
    st.warning('No trained models found. Train one or more models before running predictions.')
    st.code('python train_models.py --csv train.csv --model svm --sample 10000\nstreamlit run app.py', language='bash')
    st.stop()

with st.sidebar:
    st.header('Model selection')
    selected = st.selectbox('Choose a trained model', list(available))
    key = available[selected]
    st.caption('Only models whose saved files exist are shown.')
    if key == 'svm':
        st.info('SVM produces decision scores, not calibrated probabilities.')
    st.divider()
    st.subheader('Available trained models')
    for display in available:
        st.write('✅', display)
    st.divider()
    st.caption('Train additional models using train_models.py to unlock them here.')

comment = st.text_area('Enter an English comment', height=150, placeholder='Type a comment to classify…')

if st.button('Analyze comment', type='primary', use_container_width=True):
    if not comment.strip():
        st.warning('Please enter a comment first.')
    else:
        try:
            with st.spinner(f'Analyzing with {selected}…'):
                raw = predict(key, load_model(key), comment.strip())
                thresholds = thresholds_for(key)
                positive = raw >= thresholds
            st.subheader('Prediction result')
            cols = st.columns(2)
            cols[0].metric('Detected toxicity labels', int(positive.sum()))
            cols[1].metric('Model', selected)
            if positive.any():
                st.error('Detected: ' + ', '.join(DISPLAY[i] for i in np.where(positive)[0]))
            else:
                st.success('No toxicity labels exceeded their tuned thresholds.')
            data = pd.DataFrame({
                'Category': DISPLAY,
                'Score': raw,
                'Decision threshold': thresholds,
                'Predicted': ['Yes' if x else 'No' for x in positive],
            })
            if key == 'svm':
                st.caption('SVM scores are uncalibrated decision margins (not percentages).')
                st.dataframe(data.style.format({'Score': '{:.3f}', 'Decision threshold': '{:.3f}'}), use_container_width=True, hide_index=True)
            else:
                st.caption('The values below are sigmoid scores (0–1), not guaranteed calibrated probabilities.')
                st.bar_chart(data.set_index('Category')[['Score', 'Decision threshold']])
                st.dataframe(data.style.format({'Score': '{:.3f}', 'Decision threshold': '{:.3f}'}), use_container_width=True, hide_index=True)
            st.caption('A comment can have more than one label. Thresholds were tuned using validation data.')
        except Exception as exc:
            st.error(f'Could not run {selected}: {exc}')
            st.info('Ensure the matching model and threshold files were created successfully and required libraries are installed.')

st.divider()
st.subheader('Model comparison (Part A)')
comparison = RESULTS / 'model_comparison.csv'
if comparison.exists():
    metrics = pd.read_csv(comparison)
    if 'macro_f1' in metrics.columns:
        st.dataframe(metrics.sort_values('macro_f1', ascending=False), use_container_width=True, hide_index=True)
        st.bar_chart(metrics.set_index('model')[['macro_f1', 'micro_f1']])
    else:
        st.dataframe(metrics, use_container_width=True, hide_index=True)
else:
    st.info('Model comparison appears here after training.')
