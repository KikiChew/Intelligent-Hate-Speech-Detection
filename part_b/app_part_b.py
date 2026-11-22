"""Part B Streamlit UI: text, image OCR, CSV batch and all-model comparison.
Run from the project root: streamlit run part_b/app_part_b.py
Uses already-trained models. Does not train models in the browser.
"""
from pathlib import Path
import hashlib
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

st.set_page_config(page_title="Sentinel | Harmful Content AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
:root { --ink:#17283d; --accent:#0b88a6; }
[data-testid="stAppViewContainer"] {background:linear-gradient(145deg,#f4f8fd 0%,#fafcff 55%,#f5fafb 100%);}
[data-testid="stSidebar"] {background:#10263b;}
[data-testid="stSidebar"] * {color:#edf5fb!important;}
[data-testid="stSidebar"] [data-baseweb="select"] * {color:#14283a!important;}
[data-testid="stSidebar"] [data-baseweb="radio"] label {padding:7px 0;}
h1,h2,h3 {color:#172b42!important; letter-spacing:-.035em;}
.hero {padding:25px 30px; border-radius:22px; background:linear-gradient(115deg,#103850,#176078);color:#fff!important;margin:8px 0 24px;box-shadow:0 12px 30px rgba(28,67,92,.15);}
.hero h1 {color:#fff!important;font-size:2.0rem;margin:2px 0 10px;}
.hero p {color:#d5ebf2;margin:0;font-size:1rem;}
.hero .eyebrow {font-size:.78rem; font-weight:800; color:#7ae3e2;letter-spacing:.14em;}
[data-testid="stMetric"] {background:white;border:1px solid #e2eaf2;border-radius:16px;padding:16px;box-shadow:0 2px 12px rgba(18,38,61,.04);}
[data-testid="stMetricLabel"] {font-weight:650;color:#526478;}
[data-testid="stMetricValue"] {color:#15394d;}
[data-testid="stDataFrame"], [data-testid="stFileUploader"] {border-radius:14px;overflow:hidden;}
[data-testid="stButton"] button[kind="primary"], [data-testid="stDownloadButton"] button[kind="primary"] {border-radius:10px;background:#0c7896;border-color:#0c7896;font-weight:700;}
.small-tag {display:inline-block;font-size:.75rem; font-weight:800;border-radius:30px;padding:6px 11px;background:#def3f6;color:#086b7b;margin-bottom:10px;}
.section-note {color:#64748b;font-size:.92rem;padding:2px 0 14px;}
</style>""", unsafe_allow_html=True)
st.markdown('''<div class="hero"><div class="eyebrow">BMDS2114 · PART B v2 (ROBUST DETECTION)</div><h1>🛡️ Intelligent Hate Speech & Toxic Content Detection</h1><p>Text & image analysis · Four-model comparison · Validation metrics · Robust NLP normalization</p></div>''', unsafe_allow_html=True)

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
    for name, label in [("svm", "Linear SVM"), ("xgboost", "XGBoost"), ("textcnn", "TextCNN")]:
        model = ROOT / "part_b" / "results" / f"{name}_tuned{'.keras' if name == 'textcnn' else '.joblib'}"
        thresholds = ROOT / "part_b" / "results" / f"{name}_tuned_thresholds.csv"
        if model.is_file() and thresholds.is_file():
            models[f"{label} — Part B tuned"] = (name, model, thresholds)
    v2_model = ROOT / "part_b" / "results" / "textcnn_v2_tuned.keras"
    v2_thresholds = ROOT / "part_b" / "results" / "textcnn_v2_tuned_thresholds.csv"
    if v2_model.is_file() and v2_thresholds.is_file():
        models["TextCNN — Part B v2"] = ("textcnn", v2_model, v2_thresholds)
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
    st.metric("Toxicity categories flagged", int(predicted[0].sum()))
    frame = pd.DataFrame({"Category": LABELS, "Score": scores[0],
                          "Decision threshold": thresholds,
                          "Detected": ["Yes" if v else "No" for v in predicted[0]]})
    labels = [LABELS[i] for i in np.flatnonzero(predicted[0])]
    if labels:
        st.warning("Detected: " + ", ".join(labels))
    else:
        st.success("No toxicity label exceeded its decision threshold.")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.bar_chart(frame.set_index("Category")[["Score"]], height=230)
    if name == "svm":
        st.caption("SVM scores are decision margins, not probabilities.")
    else:
        st.caption("Model scores are not guaranteed to be calibrated probabilities.")


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

with st.sidebar:
    st.markdown("### 🛡️ SENTINEL")
    st.caption("MULTI-LABEL CONTENT SAFETY")
    page = st.radio("Explore", ["🔍 Detection Studio", "📁 Batch Analysis", "🧠 Compare All Models", "📊 Performance Dashboard"], label_visibility="collapsed")
    st.divider()
    selected = st.selectbox("Active classifier", list(OPTIONS))
    enable_normalization = st.checkbox("Robust NLP Preprocessing (v2)", value=True, help="Expands slang (e.g. u->you, fr->for real), reduces elongation (e.g. fuckassss->fuckass), removes obfuscation (e.g. f*ck->fuck), and splits compound swear words (e.g. fuckass->fuck ass).")
    st.divider()
    st.caption("Jigsaw has six toxicity labels. Identity hate is not identical to every form of hate speech. Review important decisions manually.")

if page == "🔍 Detection Studio":
    st.subheader("🔍 Detection Studio")
    st.markdown('<div class="section-note">Analyse text directly or extract text from a screenshot and correct OCR errors.</div>', unsafe_allow_html=True)
    mode = st.radio("Input type", ["Text", "Image (OCR)"], horizontal=True)
    if mode == "Text":
        text = st.text_area("Enter an English comment", height=140, key="single_text")
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

    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Enter text or extract text first.")
        else:
            raw_text = text.strip()
            processed = process_text(raw_text) if enable_normalization else raw_text
            if enable_normalization and processed != raw_text:
                st.caption("Text after robust normalization (slang expansion, elongation & compound swear segmentation):")
                st.code(processed, language=None)
            elif not enable_normalization:
                st.info("Normalization disabled: evaluating raw text directly.")
            try:
                show_details(selected, processed)
            except Exception as ex:
                st.error(f"Classification failed: {ex}")

elif page == "📁 Batch Analysis":
    st.subheader("📁 Batch Analysis")
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
                cleaned = [process_text(s) if (enable_normalization and s.strip()) else s for s in originals]
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
        st.dataframe(st.session_state["batch_result_preview"], use_container_width=True, hide_index=True)
        st.download_button("Download classified CSV", st.session_state["batch_result_csv"], "classified_comments.csv", mime="text/csv")
        st.caption("Blank comments are marked empty_text and are not classified.")

elif page == "🧠 Compare All Models":
    st.subheader("🧠 Compare All Available Models")
    st.caption("All four Part A classifiers are included when their model/threshold files are present. Trained Part B models appear automatically.")
    comment = st.text_area("Comment to compare", height=140, key="compare_text")
    if st.button("Compare Predictions", type="primary"):
        if not comment.strip():
            st.warning("Enter a comment first.")
        else:
            raw_comment = comment.strip()
            normalized = process_text(raw_comment) if enable_normalization else raw_comment
            if enable_normalization and normalized != raw_comment:
                st.write("**Text after robust normalization:**", normalized)
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
                counts = pd.DataFrame(rows)[LABELS].eq("Yes").sum()
                st.subheader("Models detecting each label")
                st.bar_chart(counts.rename("Number of models"))
                st.caption("This is model agreement, not ground-truth accuracy or a calibrated probability. SVM margins should not be compared directly with neural/XGBoost scores.")
            if errors:
                st.warning("Some models could not be loaded:\n\n" + "\n\n".join(errors))

elif page == "📊 Performance Dashboard":
    display_performance()

# The dashboard handles experiment tables and charts; keep other pages focused.
