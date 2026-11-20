# Intelligent Hate Speech and Toxic Content Detection System

**BMDS2114 Machine Learning — Group Project**  
**Part A: Model development and comparison | Part B: Model refinement (next stage)**

A multi-label NLP prototype that detects six types of toxic content in English comments using four supervised learning models. A Streamlit interface lets users select a saved model and analyse a comment. This is a research/educational prototype, **not** an automated moderation authority.

> **Terminology:** The Jigsaw labels cover several kinds of toxicity, not only hate speech. The `identity_hate` label is the dataset category most directly related to identity-based hate; the other five labels must not all be described as hate speech.

## Dataset

Source: [Jigsaw Toxic Comment Classification Challenge on Kaggle](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)

The labelled `train.csv` contains **159,571 comments** and these six binary labels:

`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

A comment can have multiple positive labels. The project's **full-data Part A** run used a reproducible **70% / 15% / 15%** split:

| Split | Samples | Usage |
|---|---:|---|
| Train | 111,699 | Fit model parameters and text representations |
| Validation | 23,936 | Select label-specific thresholds and monitor training |
| Test | 23,936 | Report held-out performance |

The split is stratified by whether a comment has **any** positive toxicity label, not by all six labels separately. Random seed: `42`.

**Dataset not included in GitHub:** Download the original labelled `train.csv` from Kaggle and place it next to `train_models.py` if you want to retrain or rerun evaluation. The Kaggle competition `test.csv` is not needed for the experiments reported here.

## Models

| Model | Feature representation | Saved checkpoint |
|---|---|---|
| Linear SVM (One-vs-Rest) | TF-IDF | `results/models/svm.joblib` |
| XGBoost (One-vs-Rest) | TF-IDF | `results/models/xgboost.joblib` |
| BiLSTM | TextVectorization + embedding | `results/models/bilstm.keras` |
| TextCNN | TextVectorization + embedding | `results/models/textcnn.keras` |

TF-IDF settings in the current script: `max_features=15000`, `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True`. The neural models use up to 20,000 tokens and a fixed sequence length of 160. All models output six category scores; predictions are determined using **per-label validation-tuned thresholds** saved in `results/<model>_thresholds.csv`.

**Important:** SVM outputs **decision margins, not probabilities**. XGBoost and neural-network scores are not necessarily calibrated probabilities.

## Part A: results (full dataset)

| Model | Validation Macro F1 | Test Macro F1 | Test Micro F1 | Test Macro ROC-AUC |
|---|---:|---:|---:|---:|
| Linear SVM | 0.5767 | 0.5820 | **0.7213** | 0.9568 |
| XGBoost | **0.5925** | **0.5864** | 0.6901 | 0.9571 |
| BiLSTM | 0.5226 | 0.5268 | 0.7051 | 0.9732 |
| TextCNN | 0.5495 | 0.5532 | 0.7117 | **0.9737** |

**Part A shortlist:** XGBoost (highest validation Macro F1) and Linear SVM (highest test Micro F1). Minority-label errors, especially `threat` and `identity_hate`, remain important limitations. Full train/validation/test metrics and per-label confusion matrices are saved under `results/`.

## Setup (Windows PowerShell)

1. Install a compatible Python version and create an isolated environment. The original development environment used Python 3.13; package compatibility may vary across machines.
2. From the project directory, create and activate an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install matplotlib
```

`matplotlib` is additionally required for the confusion-matrix output in the current training script, but **is missing from the ZIP's existing `requirements.txt`**. Add `matplotlib` to that file before committing to GitHub. Conversely, `torch` and `transformers` are legacy DistilBERT dependencies and are not used by the current four-model code; they can be removed after confirming no other scripts depend on them.

## Run Streamlit (no retraining required)

If `results/models/` and the matching threshold CSVs are included in the repository:

```powershell
streamlit run app.py
```

Open the local address shown in the terminal (usually <http://localhost:8501>). Choose Linear SVM, XGBoost, BiLSTM, or TextCNN in the sidebar, enter an English comment, and select **Analyze comment**. The app shows predicted categories and scores; it also displays the Part A comparison table.

**Never load `.joblib` or `.keras` checkpoints from untrusted sources.** Only use the project checkpoints shared by trusted teammates.

## Training commands

Download and place Kaggle's labelled `train.csv` in the root directory first.

```powershell
# Quick development run (10,000 comments)
python -X utf8 train_models.py --model all --sample 10000

# Full labelled dataset (159,571 comments)
python -X utf8 train_models.py --model all --sample 0

# Train an individual model
python -X utf8 train_models.py --model textcnn --sample 0
```

`--model all` trains the models **sequentially**, not in parallel. `-X utf8` enables Python UTF-8 mode, which helps avoid Windows encoding errors. Training the neural models on native Windows with modern TensorFlow normally runs on CPU; GPU training may require a separate supported environment such as WSL2.

**Back up `results/` before retraining.** A new run may overwrite saved checkpoints, thresholds, and metrics. Use `--out results_new` for a separate run; Streamlit currently loads the default `results/` folder.

## Evaluate saved checkpoints

The current script supports:

```powershell
python -X utf8 train_models.py --model all --sample 0 --evaluate-only
```

This regenerates Train/Validation/Test metrics and confusion matrices **without retraining**, but it requires the same labelled dataset and the original model checkpoints and thresholds. **Portability caveat:** `results/experiment_config.json` currently stores a Windows **absolute file path** and the script checks for an exact match. On a teammate's computer, `--evaluate-only` may reject an otherwise identical file because its location differs. The training/evaluation script should be updated to use a portable dataset signature or verified relative-path handling before relying on this command across machines. Do **not** bypass the check without verifying that the file content and data split are the same.

## Repository structure

```text
.
├── app.py
├── train_models.py
├── README.md
├── requirements.txt
├── .gitignore
└── results/
    ├── models/
    │   ├── svm.joblib
    │   ├── xgboost.joblib
    │   ├── bilstm.keras
    │   └── textcnn.keras
    ├── experiment_config.json
    ├── model_comparison.csv
    ├── split_metrics.csv
    ├── <model>_thresholds.csv
    ├── <model>_confusion_matrices.csv
    └── <model>_<split>_confusion_matrices.png
```

`<model>` is `svm`, `xgboost`, `bilstm`, or `textcnn`; `<split>` is `train`, `validation`, or `test`. The current ZIP also contains `*_test_scores.csv`, which are optional for sharing, and a `results (backup)/` folder from the smaller 10,000-comment pilot run. Original Kaggle CSVs, `.venv/`, caches, and backup folders should normally be excluded from Git.

## Handoff to teammate: Part B

Part A has already implemented and evaluated all four baselines. **Do not overwrite the Part A models, thresholds, or reported scores.** Suggested Part B work:

1. Tune XGBoost (`max_depth`, `learning_rate`, `n_estimators`, regularisation) using validation performance.
2. Tune Linear SVM (`C`, TF-IDF settings) to address its train–validation gap.
3. Investigate label-specific thresholds and minority-label precision/recall; optionally compare ensembles using appropriately calibrated scores.
4. Compare **baseline vs optimised** models on a consistent data split, then reserve the held-out test set for final evaluation.
5. Save all Part B work under a separate `part_b/` or `results_part_b/` directory with documented configurations.

**Note:** The saved checkpoints allow inference and baseline comparisons, but changing training hyperparameters usually requires retraining. The original Kaggle labelled data is needed for new training runs.

## Limitations and responsible use

This system was trained on English Wikipedia discussion comments and may not generalise reliably to different platforms, languages, slang, sarcasm, or changing social contexts. Minority labels are difficult to detect reliably. False positives can unfairly flag harmless comments, while false negatives can miss harmful content. Predictions should support human review rather than automatically restrict users.
