# Phase 1 — Dataset Quality & EDA

## Prepared splits
- **train**: 24,938 comments
- **validation**: 2,686 comments
- **test**: 2,687 comments
- **mybully_external_validation**: 1,159 comments

## Source and label cautions
- MYBully `Hate` and bilingual `label` may follow different annotation policies. Do not present combined binary labels as fully harmonized.
- MYBully original validation is kept separate and checked for duplicate overlap against training.
- The bilingual dataset contains heterogeneous sources, including non-hate-only sources; review per-source evaluation before claiming robustness.
- Manglish train/test have sentiment/emotion labels, not hate-speech labels; they were **not** used to train the hate classifier.
- Source metadata and MYBully manual/HITL annotation types are kept in the output CSV.
- Existing rows are deduplicated by NFKC/casefold/whitespace; more advanced near-duplicate review may still be needed.
- Chinese and tri-language Rojak coverage is **not** established by these datasets.

## Generated figures
- `hate_class_distribution.png`
- `manglish_language_distribution.png`

## Label distribution
```
                      Split    Label  Count
                      Train Non-hate  13490
                      Train     Hate  11448
                 Validation Non-hate   1453
                 Validation     Hate   1233
                       Test Non-hate   1453
                       Test     Hate   1234
MYBully external validation Non-hate    625
MYBully external validation     Hate    534
```

## Bilingual source breakdown (all splits)
```
source_detail         hate_label
HateM                 0             2078
                      1             1386
HateXplain            0             7477
                      1             6128
MYBully               0             2491
                      1             2118
Snapshot              0             4449
Supervised_sentiment  0               91
Toxicity_Small        0              435
                      1             4817
```

## Next step
Train a character/word TF-IDF + Linear SVM binary baseline and report macro F1 separately for bilingual test and MYBully external validation. Select settings only on validation data.