"""Part B Streamlit UI: text, image OCR, CSV batch and all-model comparison.
Run from the project root: streamlit run part_b/app_part_b.py
Uses already-trained models. Does not train models in the browser.
"""
from pathlib import Path
import hashlib
import html
import io
import sys
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part_b.common import LABELS, read_thresholds, score_model
from part_b.slang import normalize_slang

# Keras TextVectorization vocabulary files may require UTF-8 on Windows.
# Start with: python -X utf8 -m streamlit run part_b/app_part_b.py

st.set_page_config(page_title="Sentinel AI | Content Safety Studio", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
:root {color-scheme:light;--mint:#0b936e;--mint-dark:#075b49;--mint-tint:#eaf9f1;--ink:#173a35;--muted:#668079;--border:#dcece4;--canvas:#f7fbf8;}
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background:radial-gradient(ellipse at 98% 0%,#e3f7eb 0%,transparent 35%),radial-gradient(ellipse at 0% 95%,#f1f9eb 0%,transparent 40%),#f7fbf8;color:#173a35;}
[data-testid="stHeader"] {background:rgba(247,251,248,.88);backdrop-filter:blur(12px);}
[data-testid="stSidebar"] {background:#ffffff;border-right:1px solid #dcece4;}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span {color:#31534c;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {padding:9px 10px;margin:2px 0;border-radius:14px;transition:background .2s ease,transform .2s ease;}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background:#eaf9f1;transform:translateX(2px);}
[data-testid="stSidebar"] hr {border-color:#e5f0e9;}
h1,h2,h3,h4 {color:#163d32!important;font-family:'Plus Jakarta Sans','DM Sans',sans-serif;letter-spacing:-.035em;}
p,li,.stMarkdown {color:#31554c;}
[data-testid="stCaptionContainer"] p {color:#648078!important;}
[data-testid="stMetric"] {background:#fff;border:1px solid #dceee5;border-radius:21px;padding:19px 19px;box-shadow:0 7px 22px rgba(25,92,65,.045);transition:transform .2s ease,box-shadow .2s ease;}
[data-testid="stMetric"]:hover {transform:translateY(-3px);box-shadow:0 13px 28px rgba(25,92,65,.11);}
[data-testid="stMetricLabel"] {color:#65877b!important;font-weight:650;}
[data-testid="stMetricValue"] {color:#123c31!important;font-family:'Plus Jakarta Sans',sans-serif;}
[data-testid="stMetricDelta"] {color:#0b936e!important;}
[data-testid="stFileUploader"],[data-testid="stDataFrame"],div[data-testid="stExpander"] {border-radius:16px;overflow:hidden;}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button {border-radius:14px;min-height:43px;font-weight:700;border:1px solid #cfe7da;background:#fff;color:#146750;box-shadow:0 2px 6px #1e6d4a0b;transition:all .2s ease;}
[data-testid="stButton"] button:hover,[data-testid="stDownloadButton"] button:hover {border-color:#4dbb90;background:#eefaf3;color:#075b49;transform:translateY(-2px);box-shadow:0 6px 14px #1e6d4a20;}
[data-testid="stButton"] button[kind="primary"], [data-testid="stDownloadButton"] button[kind="primary"] {background:linear-gradient(105deg,#10a879,#08795f);border:1px solid #0a936c;color:white;box-shadow:0 6px 16px #0c996c28;}
[data-testid="stButton"] button[kind="primary"]:hover,[data-testid="stDownloadButton"] button[kind="primary"]:hover {background:linear-gradient(105deg,#07996c,#056b52);color:white;}
[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input {border-radius:15px!important;background:#fff!important;color:#183e34!important;border-color:#d7eadd!important;}
[data-testid="stTextArea"] textarea:focus,[data-testid="stTextInput"] input:focus {border-color:#1ba87d!important;box-shadow:0 0 0 2px #b8edd5!important;}
[data-testid="stSelectbox"]>div,[data-testid="stMultiSelect"]>div {border-radius:13px;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:8px;background:transparent;}
[data-testid="stTabs"] button {border-radius:12px 12px 0 0;color:#416d5e;}
[data-testid="stTabs"] button[aria-selected="true"] {color:#087e5b;border-bottom-color:#089b71;}
[data-testid="stProgressBar"]>div>div {background:#10a879;}
[data-testid="stAlert"] {border-radius:16px;}
hr {border-color:#e1eee5!important;}
.hero {padding:38px 40px;border-radius:28px;background:radial-gradient(circle at 87% 10%,#d2f6db 0%,transparent 35%),linear-gradient(112deg,#fff 0%,#f1fff6 80%);border:1px solid #d3eadb;box-shadow:0 18px 45px #235c3c0b;margin:4px 0 23px;position:relative;overflow:hidden;}
.hero:after{content:'✳';position:absolute;font-size:175px;right:3%;top:-46px;color:#b4eccc;opacity:.38;pointer-events:none;transform:rotate(12deg);}
.hero h1{font-size:2.35rem;line-height:1.18;margin:9px 0 12px;color:#133c30!important;position:relative;z-index:1;}
.hero p {color:#52786a;font-size:1.03rem;max-width:710px;margin:0;line-height:1.7;position:relative;z-index:1;}
.hero .eyebrow{color:#07875f;letter-spacing:.15em;font-weight:800;font-size:.76rem;position:relative;z-index:1;}
.pill{display:inline-block;border:1px solid #c5ead5;background:#effdf4;color:#157553;border-radius:100px;padding:7px 13px;margin:17px 7px 0 0;font-size:.8rem;font-weight:700;position:relative;z-index:1;}
.panel{border:1px solid #d9ecdf;background:#fff;border-radius:22px;padding:22px 24px;margin:8px 0 15px;box-shadow:0 9px 24px rgba(17,106,64,.05);transition:transform .2s ease,box-shadow .2s ease;}
.panel:hover{transform:translateY(-3px);box-shadow:0 12px 25px rgba(17,106,64,.10);}
.panel h3 {margin:0 0 7px;font-size:1.08rem;color:#1b5340!important;}
.panel p {color:#678077;margin:0;line-height:1.6;}
.eyebrow {color:#118461;font-size:.77rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;}
.soft-note{border:1px solid #cae9d8;background:#edfcf3;border-radius:16px;padding:14px 17px;color:#326c55;font-size:.9rem;margin:12px 0;line-height:1.6;}
.section-note {color:#688d7c;font-size:.92rem;padding:0 0 12px;}
.feature-kicker{font-size:.76rem;letter-spacing:.14em;font-weight:800;color:#118461;margin:6px 0 2px;}
.result-card{border:1px solid #d6eadd;background:#fff;border-radius:20px;padding:20px;margin:9px 0;box-shadow:0 7px 20px #1e6d4a0c;}
.result-card .status-title{font-size:.80rem;letter-spacing:.04em;text-transform:uppercase;color:#638b77;font-weight:800;}
.result-card .status-value{font-size:1.35rem;color:#153f30;font-weight:800;margin:6px 0;}
.result-card .status-subtitle{font-size:.87rem;color:#668078;}
.st-key-input_toolbar [data-testid="stButton"] button {min-height:37px;}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important;}}
@media(max-width:700px){.hero{padding:25px}.hero h1{font-size:1.73rem}.hero:after{display:none}.panel{padding:16px}}
</style>""", unsafe_allow_html=True)


def hero(eyebrow, heading, subtitle, tags=()):
    tags_html = "".join(f'<span class="pill">{html.escape(x)}</span>' for x in tags)
    st.markdown(f'<div class="hero"><div class="eyebrow">{html.escape(eyebrow)}</div>'
                f'<h1>{html.escape(heading)}</h1><p>{html.escape(subtitle)}</p>'
                f'{tags_html}</div>', unsafe_allow_html=True)


def panel(title, description, emoji="✦"):
    st.markdown(f'<div class="panel"><h3>{html.escape(emoji)} &nbsp;{html.escape(title)}</h3>'
                f'<p>{html.escape(description)}</p></div>', unsafe_allow_html=True)


def set_example(key, value):
    st.session_state[key] = value


EXAMPLES_ENGLISH = [
    ("Friendly", "Thanks for helping me today!"),
    ("Insult", "You are such an idiot and nobody likes your ideas."),
    ("Threat example", "I will hurt you if you come here again."),
]
EXAMPLES_MULTI = [
    ("English", "Thank you for helping us today!"),
    ("Bahasa Melayu", "Terima kasih sebab membantu saya."),
    ("Manglish", "Bro kau ni so annoying lah."),
    ("Malay insult", "Kau ni bodoh lah."),
]

# Exact files from Part A plus optional tuned Part B files.
def discover_models():
    models = {}
    for name, label in [("svm", "Linear SVM"), ("xgboost", "XGBoost"),
                        ("bilstm", "BiLSTM"), ("textcnn", "TextCNN")]:
        extension = ".keras" if name in ("bilstm", "textcnn") else ".joblib"
        model = ROOT / "results" / "models" / f"{name}{extension}"
        thresholds = ROOT / "results" / f"{name}_thresholds.csv"
        if model.is_file() and thresholds.is_file():
            models[f"{label} — Part A"] = (name, model, thresholds)
    for name, label in [("svm", "Linear SVM"), ("xgboost", "XGBoost")]:
        model = ROOT / "part_b" / "results" / f"{name}_tuned{'.keras' if name == 'textcnn' else '.joblib'}"
        thresholds = ROOT / "part_b" / "results" / f"{name}_tuned_thresholds.csv"
        if model.is_file() and thresholds.is_file():
            models[f"{label} — Part B tuned"] = (name, model, thresholds)
    return models

@st.cache_resource(show_spinner="Loading trained classifier...")
def load_classifier(path_string, name):
    if name in ("bilstm", "textcnn"):
        from tensorflow import keras
        return keras.models.load_model(path_string, compile=False)
    import joblib
    return joblib.load(path_string)

@st.cache_resource(show_spinner="Loading EasyOCR (first run may download weights)...")
def ocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)

@st.cache_data(show_spinner=False)
def threshold_values(path_string):
    return read_thresholds(Path(path_string))


def model_scores(name, model_path, texts):
    classifier = load_classifier(str(model_path), name)
    if name in ("svm", "xgboost"):
        scores = score_model(classifier, name, texts)
    else:
        import tensorflow as tf
        scores = classifier.predict(tf.constant(list(texts), dtype=tf.string), batch_size=64, verbose=0)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape != (len(texts), len(LABELS)):
        raise ValueError(f"Unexpected {name} scores shape {scores.shape}; expected ({len(texts)}, {len(LABELS)})")
    return scores


def evaluate_choice(choice, texts):
    name, model_path, thresholds_path = OPTIONS[choice]
    scores = model_scores(name, model_path, texts)
    thresholds = threshold_values(str(thresholds_path))
    return name, scores, thresholds, scores >= thresholds.reshape(1, -1)


def show_details(choice, text):
    name, scores, thresholds, predicted = evaluate_choice(choice, [text])
    flagcount = int(predicted[0].sum())
    headline = ", ".join(LABELS[i].replace("_", " ").title() for i in np.flatnonzero(predicted[0]))
    a, b, c = st.columns(3)
    a.metric("Categories flagged", flagcount, help="Number of model labels above their individual thresholds")
    b.metric("Categories checked", len(LABELS))
    c.metric("Model", name.upper() if name in ("svm", "xgboost") else name)
    st.progress(flagcount / len(LABELS), text=f"{flagcount} of {len(LABELS)} toxicity labels flagged (not a risk probability)")
    if flagcount:
        st.warning("⚠️ Flagged: " + headline + ". Human review recommended.")
    else:
        st.success("✅ No labels exceeded their decision thresholds. This is not a guarantee of safety.")
    frame = pd.DataFrame({"Category": [lab.replace("_", " ").title() for lab in LABELS],
                          "Score": scores[0], "Threshold": thresholds,
                          "Status": ["Flagged" if v else "Not flagged" for v in predicted[0]]})
    tab1, tab2 = st.tabs(["📈 Scores & thresholds", "📋 Detailed results"])
    with tab1:
        if name == "svm":
            st.info("SVM scores are decision margins — not probabilities. A higher margin indicates stronger model evidence.")
        else:
            st.caption("Scores are model outputs and are not necessarily calibrated probabilities.")
        st.bar_chart(frame.set_index("Category")[["Score", "Threshold"]], height=330)
    with tab2:
        st.dataframe(frame.round(4), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export detection results", frame.to_csv(index=False).encode("utf-8-sig"),
                           "sentinel_detection.csv", mime="text/csv", key="download_single_result")
    st.caption("Model predictions are experimental; review high-impact moderation decisions manually.")


def process_text(text):
    # Automatic feature. Note: Part A weights were trained on original text.
    return normalize_slang(text)

OPTIONS = discover_models()
if not OPTIONS:
    st.error("No trained model and threshold pairs found. Put part_b/ beside results/models/ in your existing project.")
    st.stop()


@st.cache_data(show_spinner=False)
def get_performance():
    rows = []
    base_file = ROOT / "results" / "split_metrics.csv"
    if base_file.exists():
        base = pd.read_csv(base_file)
        for _, r in base.iterrows():
            rows.append({"Model": str(r["model"]).upper() if str(r["model"]) in ("svm", "xgboost") else str(r["model"]).title(),
                         "Stage": "Part A baseline", "Split": str(r["split"]),
                         "Macro F1": float(r["macro_f1"]), "Micro F1": float(r["micro_f1"]),
                         "Precision": float(r["micro_precision"]), "Recall": float(r["micro_recall"]),
                         "ROC-AUC": float(r["macro_roc_auc"]) if pd.notna(r.get("macro_roc_auc")) else np.nan,
                         **{lab: float(r.get(lab+"_f1", np.nan)) for lab in LABELS}})
    # Only load tuned metrics if they exist and contain all expected splits.
    for key in ("svm", "xgboost", "textcnn"):
        path = ROOT / "part_b" / "results" / f"{key}_tuned_split_metrics.csv"
        if not path.exists():
            continue
        tab = pd.read_csv(path)
        for _, r in tab.iterrows():
            rows.append({"Model": key.upper() if key in ("svm", "xgboost") else "TextCNN",
                         "Stage": "Part B tuned", "Split": str(r["split"]),
                         "Macro F1": float(r["macro_f1"]), "Micro F1": float(r["micro_f1"]),
                         "Precision": float(r["micro_precision"]), "Recall": float(r["micro_recall"]),
                         "ROC-AUC": float(r["macro_roc_auc"]) if pd.notna(r.get("macro_roc_auc")) else np.nan,
                         **{lab: float(r.get(lab+"_f1", np.nan)) for lab in LABELS}})
    v2_file = ROOT / "part_b" / "results" / "textcnn_v2_tuned_split_metrics.csv"
    if v2_file.exists():
        for _, r in pd.read_csv(v2_file).iterrows():
            rows.append({"Model":"TextCNN", "Stage":"Part B v2", "Split":str(r['split']),
                         "Macro F1":float(r['macro_f1']), "Micro F1":float(r['micro_f1']),
                         "Precision":float(r['micro_precision']), "Recall":float(r['micro_recall']),
                         "ROC-AUC":float(r['macro_roc_auc']),
                         **{lab:float(r.get(lab+'_f1', np.nan)) for lab in LABELS}})
    return pd.DataFrame(rows)

def display_performance():
    st.subheader("📊 Model Performance Dashboard")
    st.caption("Measured experimental results loaded from your CSV files; no scores are simulated.")
    data = get_performance()
    if data.empty:
        st.warning("No performance files found. Place Part A results/split_metrics.csv in the repository.")
        return
    expected_counts = {"train":111699,"validation":23936,"test":23936}
    # Detect inadvertent mixing with the old 10k-sample experiment.
    paths = [(ROOT/'results'/'split_metrics.csv', 'Part A'),
             *((ROOT/'part_b'/'results'/f'{k}_tuned_split_metrics.csv', k) for k in ('svm','xgboost','textcnn')),
             (ROOT/'part_b'/'results'/'textcnn_v2_tuned_split_metrics.csv','TextCNN v2')]
    for p, label in paths:
        if p.exists():
            raw = pd.read_csv(p)
            if 'samples' in raw:
                inconsistent = [(str(r['split']), int(r['samples'])) for _,r in raw.iterrows()
                                if str(r['split']) in expected_counts and int(r['samples'])!=expected_counts[str(r['split'])]]
                if inconsistent:
                    st.error(f"{label} metrics do not match full-data 70/15/15 splits: {inconsistent}. Do not compare mixed sample sizes.")
                    return
    validation = data[data["Split"]=="validation"].copy()
    test = data[data["Split"]=="test"].copy()
    if len(validation):
        best = validation.loc[validation["Macro F1"].idxmax()]
        a,b,c,d=st.columns(4)
        a.metric("Best validation Macro F1",f"{best['Macro F1']:.4f}")
        b.metric("Leading model",str(best['Model']))
        c.metric("Available baselines",int(data[data['Stage']=='Part A baseline']['Model'].nunique()))
        d.metric("Tuned models evaluated",int(data[data['Stage'].str.startswith('Part B')]['Model'].nunique()))
    st.divider()
    metric = st.selectbox("Metric", ["Macro F1","Micro F1","Precision","Recall","ROC-AUC"], key="perf_metric")
    subset = st.selectbox("Evaluation split", ["validation","test","train"],key="perf_split")
    view=data[data["Split"]==subset].copy()
    view["Variant"] = view["Model"] + " · " + view["Stage"].str.replace("Part A baseline","baseline").str.replace("Part B tuned","tuned")
    st.markdown("#### Baseline and tuned model results")
    st.bar_chart(view.set_index("Variant")[[metric]], height=360)
    st.dataframe(view[["Model","Stage","Split","Macro F1","Micro F1","Precision","Recall","ROC-AUC"]].round(4), use_container_width=True, hide_index=True)
    st.markdown("#### Change after tuning · Validation vs Test")
    comparison = data[data['Split'].isin(['validation','test'])].pivot_table(
        index=['Model','Split'], columns='Stage', values='Macro F1', aggfunc='first')
    if 'Part A baseline' in comparison.columns and 'Part B tuned' in comparison.columns:
        comp = comparison[['Part A baseline','Part B tuned']].dropna().copy()
        comp['Δ F1 (B − A)'] = comp['Part B tuned'] - comp['Part A baseline']
        st.dataframe(comp.round(4), use_container_width=True)
        # Explicit columns avoid Streamlit's generated-index / pandas MultiIndex issues.
        delta_chart = comp[['Δ F1 (B − A)']].reset_index()
        delta_chart = delta_chart.rename(columns={'Δ F1 (B − A)': 'F1 Difference'})
        delta_chart['Model / Split'] = (
            delta_chart['Model'].astype(str) + ' - ' + delta_chart['Split'].astype(str)
        )
        st.bar_chart(
            delta_chart[['Model / Split', 'F1 Difference']],
            x='Model / Split',
            y='F1 Difference',
            height=300,
        )
    st.markdown("#### Generalization · Macro F1 by split")
    wide=data.assign(Variant=lambda x:x["Model"]+" · "+x["Stage"].map({"Part A baseline":"A","Part B tuned":"B", "Part B v2":"B2"})).pivot_table(index="Variant",columns="Split",values="Macro F1",aggfunc="first")
    st.bar_chart(wide.reindex(columns=["train","validation","test"]), height=330)
    st.markdown("#### Label-level performance")
    options=sorted(view["Variant"].unique().tolist())
    if options:
        choice=st.selectbox("Choose a model for per-label F1",options)
        r=view[view["Variant"]==choice].iloc[0]
        by_label=pd.DataFrame({"F1": [r[k] for k in LABELS]}, index=[x.replace('_',' ').title() for x in LABELS])
        st.bar_chart(by_label,height=290)
    st.divider()
    st.markdown("#### Confusion matrix explorer")
    label_choice=st.selectbox("Toxicity label",LABELS)
    matrix_choice=st.selectbox("Model", sorted(data['Model'].unique()))
    stage_choice=st.radio("Version",["Part A baseline","Part B tuned","Part B v2"],horizontal=True)
    if stage_choice=='Part A baseline':
        path=ROOT/'results'/f"{matrix_choice.lower()}_confusion_matrices.csv"
    elif stage_choice == 'Part B tuned':
        path=ROOT/'part_b'/'results'/f"{matrix_choice.lower()}_tuned_confusion.csv"
    else:
        path=ROOT/'part_b'/'results'/'textcnn_v2_tuned_confusion.csv' if matrix_choice == 'TextCNN' else ROOT/'missing.csv'
    if path.exists():
        cm=pd.read_csv(path)
        if 'split' in cm.columns: cm=cm[cm['split']==subset]
        if 'label' in cm.columns: cm=cm[cm['label']==label_choice]
        if not cm.empty:
            rr=cm.iloc[0]
            vals={key:int(rr[key]) for key in ['TN','FP','FN','TP']}
            aa,bb,cc,dd=st.columns(4)
            aa.metric('True negatives',vals['TN']);bb.metric('False positives',vals['FP'])
            cc.metric('False negatives',vals['FN']);dd.metric('True positives',vals['TP'])
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            fig,ax=plt.subplots(figsize=(4.8,3.35))
            grid=np.array([[vals['TN'], vals['FP']],[vals['FN'], vals['TP']]])
            ax.imshow(grid,cmap=LinearSegmentedColormap.from_list('tealwhite',['#e6f3f8','#125f7c']))
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,f"{grid[i,j]:,}",ha='center',va='center',fontsize=15,fontweight='bold',color='#132b40')
            ax.set_xticks([0,1],['Predicted 0','Predicted 1']);ax.set_yticks([0,1],['Actual 0','Actual 1']);ax.set_title(label_choice.replace('_',' ').title())
            fig.tight_layout();st.pyplot(fig,clear_figure=True);plt.close(fig)
        else:
            st.info('No confusion counts available for this combination.')
    else:
        st.info('Confusion matrix CSV not yet available for this model/version. Train or evaluate it first.')
    st.divider()
    st.markdown("#### Hyperparameter search history")
    for model_key in ['svm','xgboost','textcnn','textcnn_v2']:
        file=ROOT/'part_b'/'results'/f'{model_key}_search.csv'
        if file.exists():
            with st.expander(f"{model_key.upper()} tuning candidates",expanded=False):
                tab=pd.read_csv(file)
                st.dataframe(tab.round(4),use_container_width=True,hide_index=True)
                if 'macro_f1' in tab:
                    chart=tab[['candidate','macro_f1']].copy()
                    chart['candidate']=chart['candidate'].astype(str)
                    st.bar_chart(chart.set_index('candidate'),height=220)
    st.markdown("#### TextCNN training curves")
    history_options = {
        'TextCNN v1':ROOT/'part_b'/'results'/'textcnn_tuned_training_history.csv',
        'TextCNN v2':ROOT/'part_b'/'results'/'textcnn_v2_tuned_training_history.csv'
    }
    for title, f in history_options.items():
        if f.exists():
            with st.expander(title, expanded=False):
                hist = pd.read_csv(f)
                if {'loss','val_loss'}.issubset(hist.columns):
                    st.line_chart(hist[['loss','val_loss']], height=240)
                else:
                    st.dataframe(hist,hide_index=True)
    st.caption('Validation is used for selecting hyperparameters. Avoid repeated model selection on the test set.')


# --- Independent binary multilingual workflow; distinct from Jigsaw six labels. ---
MULTI_ROOT = ROOT / "part_b" / "multilingual" / "results"
XLMR_FULL = MULTI_ROOT / "xlmr_multilingual_full"
XLMR_OLD = MULTI_ROOT / "xlmr_multilingual"
MULTI_SVM_PATH = MULTI_ROOT / "svm_multilingual" / "multilingual_svm.joblib"


def multilingual_availability():
    options = {}
    if MULTI_SVM_PATH.is_file():
        options["Multilingual SVM"] = "svm"
    if (XLMR_FULL / "model" / "config.json").is_file():
        options["XLM-RoBERTa (full training)"] = "xlmr"
    return options


@st.cache_resource(show_spinner="Loading multilingual SVM...")
def get_multi_svm():
    from part_b.multilingual.predict_svm import load_model
    return load_model(MULTI_SVM_PATH)


@st.cache_resource(show_spinner="Loading full-data XLM-RoBERTa...")
def get_multi_xlmr():
    from part_b.multilingual.predict_xlmr import load_model
    # Inference defaults to CPU to avoid contention with TF/EasyOCR.
    return load_model(XLMR_FULL / "model")


def multilingual_predict(text, selection):
    if selection == "svm":
        from part_b.multilingual.predict_svm import predict
        return predict([text], get_multi_svm())[0]
    from part_b.multilingual.predict_xlmr import predict
    return predict(text, get_multi_xlmr(), max_length=64)


def read_multi_metrics():
    rows = []
    for model, location in [
        ("Multilingual SVM", MULTI_ROOT / "svm_multilingual" / "svm_split_metrics.csv"),
        ("XLM-RoBERTa", XLMR_FULL / "split_metrics.csv"),
    ]:
        if location.exists():
            try:
                tab = pd.read_csv(location)
                if {"split", "macro_f1"}.issubset(tab.columns):
                    for _, r in tab.iterrows():
                        rows.append({"Model": model, "Split": str(r["split"]),
                                     "Macro F1": float(r["macro_f1"]),
                                     "Accuracy": float(r["accuracy"]) if "accuracy" in r else np.nan,
                                     "ROC-AUC": float(r["roc_auc"]) if "roc_auc" in r else np.nan})
            except Exception as ex:
                st.warning(f"Could not read metrics from {location.name}: {ex}")
    return pd.DataFrame(rows)


def display_multilingual():
    hero("MULTILINGUAL INTELLIGENCE", "🌏 Language-aware detection",
         "Analyze English, Bahasa Melayu, and experimental Manglish using two independently trained binary hate-speech models.",
         ("English + Malay", "SVM vs XLM-R", "Binary Hate / Non-Hate"))
    available = multilingual_availability()
    if not available:
        st.warning("No multilingual model found. Check part_b/multilingual/results/.")
    st.markdown("#### Try a sample")
    cols = st.columns(4)
    for col, (label, example) in zip(cols, EXAMPLES_MULTI):
        col.button(label, on_click=set_example, args=("multi_input", example),
                   key="try_multi_"+label.replace(" ", "_"), use_container_width=True)
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        text = st.text_area("Comment to analyze", value="Bro kau ni so annoying lah.", height=160,
                            key="multi_input", placeholder="Enter English, BM or Manglish comment...")
        st.caption(f"{len(text):,} characters · Input language is not automatically verified")
        # Multilingual models were trained/evaluated using their established preprocessing.
        # Keep slang normalization opt-in to preserve the default inference behavior.
        multi_norm = st.toggle("🧹 Normalize chat slang (Kaggle dataset)", value=False, key="multi_slang_toggle", help="Optional text expansion. May change predictions; not part of the reported test metrics.")
        multi_text = text.strip()
        if multi_norm and multi_text:
            try:
                multi_text = normalize_slang(multi_text)
            except (FileNotFoundError, ValueError) as ex:
                st.error(f"Slang data unavailable: {ex}")
                multi_text = text.strip()
        with st.expander("👀 Original vs normalized preview", expanded=multi_norm):
            st.caption("Dataset: Kaggle Chat Slang Abbreviations Acronyms · part_b/data/slang/slang.json")
            st.write("**Original:**", text)
            st.write("**Sent to models:**", multi_text)
            if multi_text == text.strip():
                st.caption("No change (or normalization is switched off).")
    with right:
        panel("Choose your analysis", "Compare both models for added context, or select one for a faster prediction.", "🧠")
        compare = st.toggle("Compare both multilingual models", value=True, disabled=len(available)<2)
        chosen = st.selectbox("Primary model", list(available), disabled=not bool(available)) if available else None
        if not compare:
            st.caption("Only the chosen model will run.")
    if st.button("🌱 Analyze comment", type="primary", disabled=not available,
                 key="run_multilingual", use_container_width=True):
        if not text.strip():
            st.warning("Enter a comment first.")
        else:
            targets = list(available) if compare and len(available)>1 else [chosen]
            results = []
            for model_name in targets:
                try:
                    with st.spinner(f"Analyzing with {model_name}..."):
                        prediction = multilingual_predict(multi_text, available[model_name])
                    entry = {"Model": model_name, "Prediction": prediction["label"]}
                    if "hate_probability" in prediction:
                        entry["Hate probability (uncalibrated)"] = prediction["hate_probability"]
                    if "decision_score" in prediction:
                        entry["SVM decision margin"] = prediction["decision_score"]
                    results.append(entry)
                except Exception as ex:
                    st.error(f"{model_name} failed: {type(ex).__name__}: {ex}")
            st.session_state["multi_last_result"] = {"text": text.strip(), "normalized": multi_text, "rows": results}
    stored = st.session_state.get("multi_last_result")
    if stored and stored.get("rows"):
        if stored["text"] != text.strip() or stored.get("normalized", stored["text"]) != multi_text:
            st.info("Showing the previous result. Click Analyze comment to analyze the current text and normalization setting.")
        st.markdown("### 🔎 Detection results")
        st.caption(f"Input used for prediction: {stored.get('normalized', stored['text'])}")
        with st.expander("ℹ️ Hate speech vs personal insult — how to read this result", expanded=False):
            st.markdown("**Personal insult / toxicity** targets an individual (e.g., ‘you are stupid’). **Identity-based hate** targets people based on characteristics such as ethnicity or religion. These categories can overlap, but they are not equivalent. The multilingual models output only **Hate / Non-Hate**; they do **not** separately classify insults. Since the training datasets include different definitions of harmful content, a Non-Hate prediction is **not** a guarantee that a comment is harmless. Review ambiguous or discriminatory content manually.")

        table = pd.DataFrame(stored["rows"])
        cols = st.columns(len(table))
        for col, (_, r) in zip(cols, table.iterrows()):
            col.metric(str(r["Model"]), str(r["Prediction"]),
                       help="Experimental prediction; not a final human moderation decision")
            if pd.notna(r.get("Hate probability (uncalibrated)", np.nan)):
                probability = float(r["Hate probability (uncalibrated)"])
                col.progress(min(100, max(0, round(probability*100))), text=f"Hate score: {probability:.1%}")
                col.caption("Uncalibrated model probability")
            if pd.notna(r.get("SVM decision margin", np.nan)):
                col.caption(f"SVM decision margin: {float(r['SVM decision margin']):.3f} (not a probability)")
        if len(table) >= 2:
            verdicts = table["Prediction"].astype(str).str.casefold().unique()
            if len(verdicts) == 1:
                st.success("✅ Both models agree on the prediction. Agreement is not proof of correctness.")
            else:
                st.warning("🔎 Models disagree. Consider manual review before acting on this prediction.")
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Export multilingual prediction", table.to_csv(index=False).encode("utf-8-sig"),
                           "multilingual_predictions.csv", mime="text/csv", key="download_multi")
        st.caption("SVM margin cannot be compared numerically to XLM-R probabilities. Disagreements should be reviewed by a person.")
    st.divider()
    st.markdown("### 📊 Multilingual evaluation lab")
    metrics = read_multi_metrics()
    if not metrics.empty:
        splits = [s for s in ["validation", "test", "mybully_external"] if s in metrics["Split"].values]
        selected_split = st.radio("Evaluation dataset", splits, index=splits.index("test") if "test" in splits else 0,
                                  horizontal=True, key="multi_eval_split")
        split_data = metrics[metrics["Split"]==selected_split]
        if not split_data.empty:
            c1, c2 = st.columns([1.6, 1], gap="large")
            with c1:
                st.bar_chart(split_data[["Model", "Macro F1"]], x="Model", y="Macro F1", height=275)
            with c2:
                for _, row in split_data.iterrows():
                    st.metric(row["Model"], f"{row['Macro F1']:.4f}", help="Macro F1 on selected dataset")
                st.caption("Macro F1 treats both classes equally.")
            with st.expander("View raw evaluation metrics"):
                st.dataframe(split_data.round(4), hide_index=True, use_container_width=True)
    else:
        st.info("Model evaluation files are not available in results folders.")
    one, two = st.tabs(["🗣️ Performance by language", "🧩 Confusion matrix"])
    with one:
        lang_file = XLMR_FULL / "language_metrics.csv"
        if lang_file.is_file():
            language_data = pd.read_csv(lang_file)
            if {"split", "language", "macro_f1"}.issubset(language_data.columns):
                test_lang = language_data[language_data["split"]=="test"]
                if not test_lang.empty:
                    st.bar_chart(test_lang[["language", "macro_f1"]], x="language", y="macro_f1", height=260)
                    st.dataframe(test_lang.round(4), hide_index=True, use_container_width=True)
        else:
            st.info("Language metrics file is missing.")
    with two:
        cm_file = XLMR_FULL / "confusion_matrix.csv"
        if cm_file.is_file():
            table = pd.read_csv(cm_file)
            if {"split", "tn", "fp", "fn", "tp"}.issubset(table.columns):
                chosen_split = st.selectbox("Matrix dataset", list(table["split"].unique()),
                                            index=list(table["split"]).index("test") if "test" in table["split"].values else 0)
                rr = table[table["split"]==chosen_split].iloc[0]
                cols = st.columns(4)
                for col, label, key in zip(cols, ["True negatives", "False positives", "False negatives", "True positives"], ["tn", "fp", "fn", "tp"]):
                    col.metric(label, int(rr[key]))
                st.caption("False positives = safe comments flagged; false negatives = harmful comments missed.")
                with st.expander("View source confusion matrix"):
                    st.dataframe(table, hide_index=True, use_container_width=True)
        else:
            st.info("Confusion matrix file is missing.")
    st.caption("Results validate labelled datasets only; real-world Manglish/Chinese code-mixing requires further evaluation.")

NAV = ["🏠 Overview", "🔍 Detection Studio", "🌏 Multilingual Detection", "📁 Batch Analysis", "🧠 Compare All Models", "📊 Performance Dashboard"]
with st.sidebar:
    st.markdown("## 🌿 SENTINEL AI")
    st.caption("Mint Intelligence Edition")
    st.caption("CONTENT SAFETY · RESEARCH STUDIO")
    st.divider()
    page = st.radio("Navigation", NAV, key="navigation", label_visibility="collapsed")
    st.divider()
    st.markdown("**🔬 English six-label classifier**")
    selected = st.selectbox("Active model", list(OPTIONS), key="active_classifier")
    st.caption("Applied to English Detection, Batch and Model Comparison pages.")
    st.divider()
    st.markdown("**⚙️ System status**")
    st.caption(f"✅ {len(OPTIONS)} English model variants available")
    st.caption(f"✅ {len(multilingual_availability())} multilingual models available" if multilingual_availability() else "⚠️ Multilingual models not found")
    st.divider()
    st.caption("Academic prototype · Not a substitute for human moderation")

if page == "🏠 Overview":
    hero("AI-POWERED CONTENT SAFETY", "Make safer conversations possible.",
         "One workspace for multilingual hate-speech detection, six-label toxicity analysis, image OCR and transparent model evaluation.",
         ("6 toxicity categories", "English · Malay · Manglish", "Explainable results"))
    a, b, c, d = st.columns(4)
    a.metric("English models", len(OPTIONS))
    b.metric("Multilingual models", len(multilingual_availability()))
    c.metric("Toxicity categories", len(LABELS))
    metrics = read_multi_metrics()
    best_test = metrics[(metrics["Split"] == "test") & (metrics["Model"] == "XLM-RoBERTa")] if not metrics.empty else pd.DataFrame()
    d.metric("XLM-R Test Macro F1", f"{best_test.iloc[0]['Macro F1']:.3f}" if not best_test.empty else "—")
    st.markdown("### 🚀 Choose a workflow")
    cards = [
        ("🔍 Analyze a comment", "English six-category toxicity detection with label-level evidence.", "🔍 Detection Studio"),
        ("🌏 Detect across languages", "English, Bahasa Melayu and experimental Manglish, with model comparison.", "🌏 Multilingual Detection"),
        ("📁 Review a CSV", "Classify up to 500 comments and export a structured report.", "📁 Batch Analysis"),
        ("🧠 Compare classifiers", "Inspect agreements and disagreements across available English models.", "🧠 Compare All Models"),
    ]
    columns = st.columns(2, gap="large")
    for i, (name, desc, dest) in enumerate(cards):
        with columns[i % 2]:
            panel(name, desc, "")
            st.button("Open " + name.split(' ', 1)[1], key=f"home_{i}", use_container_width=True,
                      on_click=set_example, args=("navigation", dest))
    st.markdown("### 📈 Research snapshot")
    if not metrics.empty and "test" in metrics["Split"].values:
        st.bar_chart(metrics.loc[metrics["Split"] == "test", ["Model", "Macro F1"]], x="Model", y="Macro F1", height=270)
    st.markdown('<div class="soft-note">🧪 <b>Research note:</b> English toxicity detection (six labels) and multilingual binary hate detection solve different tasks. Their metrics should not be compared as if they were the same benchmark.</div>', unsafe_allow_html=True)

if page == "🔍 Detection Studio":
    hero("ENGLISH TOXICITY LAB", "🔍 Detection studio", "Analyze English comments across six toxicity categories. You can also extract text from a screenshot and correct it before prediction.", ("6 labels", "OCR supported", "Downloadable results"))
    mode = st.radio("Input type", ["Text", "Image (OCR)"], horizontal=True)
    if mode == "Text":
        st.markdown("**✨ Quick-fill examples**")
        example_cols = st.columns(3)
        for col, (label, example) in zip(example_cols, EXAMPLES_ENGLISH):
            col.button(label, key="example_en_"+label, on_click=set_example, args=("single_text", example), use_container_width=True)
        with st.expander("💡 Input tips & shortcuts", expanded=False):
            st.write("Select a sample above or paste a comment. Normalization happens when you run analysis. OCR mode lets you correct extracted text first.")
        text = st.text_area("Enter an English comment", height=160, key="single_text", placeholder="Type or paste an English comment here...")
        st.caption(f"{len(text):,} characters · Current model: {selected}")
    else:
        uploaded = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="single_image")
        if uploaded is not None:
            from PIL import Image, UnidentifiedImageError
            blob = uploaded.getvalue()
            digest = hashlib.sha256(blob).hexdigest()
            if st.session_state.get("image_digest") != digest:
                st.session_state["image_digest"] = digest
                st.session_state["ocr_editable"] = ""
                st.session_state["ocr_confidence"] = None
                st.session_state["ocr_items"] = 0
            try:
                picture = Image.open(io.BytesIO(blob)).convert("RGB")
                st.image(picture, caption="Uploaded image", width=500)
                if st.button("Extract Text", type="secondary"):
                    with st.spinner("Extracting text..."):
                        # easyocr returns (bounding box, text, confidence)
                        entries = ocr_reader().readtext(np.asarray(picture), detail=1, paragraph=False)
                    parts = [word for _, word, _ in entries]
                    confidence = [float(score) for _, _, score in entries]
                    st.session_state["ocr_editable"] = " ".join(parts)
                    st.session_state["ocr_confidence"] = float(np.mean(confidence)) if confidence else None
                    st.session_state["ocr_items"] = len(parts)
            except ImportError:
                st.error("Install OCR dependencies with: pip install -r part_b/requirements_ocr.txt")
            except (OSError, UnidentifiedImageError, ValueError) as ex:
                st.error(f"Could not read image: {ex}")
            except Exception as ex:
                st.error(f"OCR failed: {ex}")
        else:
            st.session_state["ocr_editable"] = ""
            st.session_state["ocr_confidence"] = None
        conf = st.session_state.get("ocr_confidence")
        if conf is not None:
            st.metric("Average OCR confidence (not classification confidence)", f"{conf:.1%}")
            st.caption(f"Recognized text regions: {st.session_state.get('ocr_items', 0)}. Confidence is a rough OCR estimate, not accuracy.")
        st.caption("Review and correct any OCR mistakes before classification.")
        text = st.text_area("Editable extracted text", key="ocr_editable", height=150)

    english_norm = st.toggle("🧹 Enable Kaggle slang normalization", value=True, key="english_slang_toggle", help="Expand chat abbreviations before prediction; the model was not necessarily trained on the expanded text.")
    if text.strip():
        try:
            english_processed_preview = process_text(text.strip()) if english_norm else text.strip()
        except (FileNotFoundError, ValueError) as ex:
            st.error(f"Slang dataset could not load: {ex}")
            english_processed_preview = text.strip()
        with st.expander("👀 Original vs normalized preview", expanded=False):
            st.caption("Dataset: Kaggle Chat Slang Abbreviations Acronyms")
            st.write("**Original:**", text.strip())
            st.write("**Sent to model:**", english_processed_preview)
    if st.button("🌱 Analyze comment", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Enter text or extract text first.")
        else:
            st.session_state["single_result"] = (selected, text.strip(), english_processed_preview)
    record = st.session_state.get("single_result")
    if record:
        result_model, original, processed = record
        if original != text.strip() or result_model != selected:
            st.info("Showing your previous analysis. Click Analyze toxicity to run on the current input and model.")
        if processed != original:
            with st.expander("🧹 Show normalized text"):
                st.code(processed, language=None)
        try:
            show_details(result_model, processed)
        except Exception as ex:
            st.error(f"Classification failed: {ex}")

elif page == "📁 Batch Analysis":
    hero("BULK CONTENT REVIEW", "📁 Batch analysis", "Upload a CSV, select a text column, inspect model flags and export categorized results.", ("Up to 500 comments", "CSV export", "Progress tracking"))
    st.write("Upload a UTF-8 CSV with a **comment_text** column (or choose another text column). Up to 500 rows per run.")
    st.download_button("Download CSV template", "comment_text\nThank you for your help!\nu are rude\n", "batch_template.csv", "text/csv")
    file = st.file_uploader("CSV file", type=["csv"], key="batch_file")
    if file is not None:
        try:
            if file.size > 5 * 1024 * 1024:
                st.error("Please upload a CSV smaller than 5 MB.")
                st.stop()
            df = pd.read_csv(file)
            if len(df) == 0:
                st.error("The CSV has no rows.")
                st.stop()
            st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
            column = st.selectbox("Text column", df.columns, index=list(df.columns).index("comment_text") if "comment_text" in df.columns else 0)
            st.dataframe(df.head(5), use_container_width=True, hide_index=True)
            if len(df) > 500:
                st.warning("More than 500 rows found. Only the first 500 will be processed. Split your file for larger batches.")
            signature = (hashlib.sha256(file.getvalue()).hexdigest(), selected, column)
            if st.session_state.get("batch_signature") != signature:
                st.session_state["batch_signature"] = signature
                for state_key in ("batch_result_csv", "batch_result_preview", "batch_result_count"):
                    st.session_state.pop(state_key, None)
            if st.button("Classify CSV", type="primary"):
                batch = df.head(500).copy()
                originals = batch[column].fillna("").astype(str).tolist()
                cleaned = [process_text(s) if s.strip() else "" for s in originals]
                labels_pred = np.zeros((len(cleaned), len(LABELS)), dtype=bool)
                valid = [i for i, s in enumerate(cleaned) if s.strip()]
                progress = st.progress(0, text="Starting batch prediction...")
                try:
                    for step, start in enumerate(range(0, len(valid), 64)):
                        ids = valid[start:start + 64]
                        _, _, _, pred = evaluate_choice(selected, [cleaned[i] for i in ids])
                        labels_pred[ids, :] = pred
                        progress.progress(min(1.0, (start + len(ids)) / max(1, len(valid))), text=f"Processed {min(start + len(ids), len(valid))}/{len(valid)} nonempty rows")
                    progress.empty()
                    out = batch.copy()
                    out["normalized_text"] = cleaned
                    out["prediction_status"] = ["ok" if i in set(valid) else "empty_text" for i in range(len(cleaned))]
                    for j, label in enumerate(LABELS):
                        out[f"pred_{label}"] = labels_pred[:, j].astype(int)
                    out["detected_labels"] = [", ".join([LABELS[j] for j in np.flatnonzero(row)]) for row in labels_pred]
                    out["detected_count"] = labels_pred.sum(axis=1)
                    st.session_state["batch_result_csv"] = out.to_csv(index=False).encode("utf-8-sig")
                    st.session_state["batch_result_preview"] = out.head(30)
                    st.session_state["batch_result_count"] = len(batch)
                except Exception as ex:
                    progress.empty()
                    st.error(f"Batch classification failed: {ex}")
        except (UnicodeDecodeError, pd.errors.ParserError) as ex:
            st.error(f"Invalid CSV or encoding: {ex}. Save as UTF-8 CSV.")
    if "batch_result_csv" in st.session_state:
        st.success(f"Processed {st.session_state['batch_result_count']} comments. Results are ready for download.")
        preview = st.session_state["batch_result_preview"]
        a, b, c = st.columns(3)
        a.metric("Preview rows", len(preview))
        b.metric("Flagged (preview)", int((preview["detected_count"] > 0).sum()))
        c.metric("No flags (preview)", int((preview["detected_count"] == 0).sum()))
        view_filter = st.radio("Filter preview", ["All", "Flagged", "No flags"], horizontal=True, key="batch_preview_filter")
        filtered = preview if view_filter == "All" else preview[preview["detected_count"] > 0]
        if view_filter == "No flags":
            filtered = preview[preview["detected_count"] == 0]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.caption("Filter only affects the 30-row preview. Export always contains the full batch.")
        st.download_button("Download classified CSV", st.session_state["batch_result_csv"], "classified_comments.csv", mime="text/csv")
        st.caption("Blank comments are marked empty_text and are not classified.")

elif page == "🧠 Compare All Models":
    hero("MODEL CONSENSUS LAB", "🧠 Compare classifiers", "Discover where trained English toxicity models agree and disagree, category by category.", ("Side-by-side", "Six labels", "Transparent scores"))
    st.caption("All four Part A classifiers are included when their model/threshold files are present. Trained Part B models appear automatically.")
    comment = st.text_area("Comment to compare", height=140, key="compare_text")
    if st.button("Compare Predictions", type="primary"):
        if not comment.strip():
            st.warning("Enter a comment first.")
        else:
            normalized = process_text(comment.strip())
            if normalized != comment.strip():
                st.write("**Automatically processed text:**", normalized)
            rows = []
            errors = []
            for selection in OPTIONS:
                try:
                    name, scores, thresholds, flags = evaluate_choice(selection, [normalized])
                    row = {"Model": selection, "Number of labels": int(flags[0].sum())}
                    row.update({label: "Yes" if flags[0, j] else "No" for j, label in enumerate(LABELS)})
                    rows.append(row)
                except Exception as ex:
                    errors.append(f"{selection}: {ex}")
            if rows:
                table = pd.DataFrame(rows).set_index("Model")
                st.dataframe(table, use_container_width=True)
                st.download_button("⬇️ Export comparison CSV", table.to_csv().encode("utf-8-sig"), "model_comparison_predictions.csv", mime="text/csv")
                counts = pd.DataFrame(rows)[LABELS].eq("Yes").sum()
                st.subheader("Models detecting each label")
                st.bar_chart(counts.rename("Number of models"))
                st.caption("This is model agreement, not ground-truth accuracy or a calibrated probability. SVM margins should not be compared directly with neural/XGBoost scores.")
            if errors:
                st.warning("Some models could not be loaded:\n\n" + "\n\n".join(errors))

elif page == "📊 Performance Dashboard":
    hero("EXPERIMENT EXPLORER", "📊 Model performance", "Compare baseline and tuned English classifiers using recorded evaluation metrics, training curves and confusion matrices.", ("Real CSV metrics", "Validation / test", "Per-label insights"))
    display_performance()

elif page == "🌏 Multilingual Detection":
    display_multilingual()

# The dashboard handles experiment tables and charts; keep other pages focused.
