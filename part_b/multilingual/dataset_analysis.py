"""Generate charts and plain-language summary from cleaned multilingual CSVs."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE/'results'

def main():
    data = []
    for name,file in [('Train','train_clean.csv'),('Validation','val_clean.csv'),
                      ('Test','test_clean.csv'),('MYBully external validation','mybully_external_val.csv')]:
        d=pd.read_csv(OUT/file)
        data.extend([{'Split':name,'Label':'Non-hate','Count':int((d.hate_label==0).sum())},
                     {'Split':name,'Label':'Hate','Count':int((d.hate_label==1).sum())}])
    table=pd.DataFrame(data)
    pivot=table.pivot(index='Split',columns='Label',values='Count').fillna(0)
    fig,ax=plt.subplots(figsize=(10,4.7))
    pivot.plot(kind='bar',stacked=True,ax=ax)
    ax.set_title('Hate-label distribution by split')
    ax.set_ylabel('Number of comments')
    ax.set_xlabel('')
    ax.tick_params(axis='x',rotation=15)
    fig.tight_layout()
    fig.savefig(OUT/'hate_class_distribution.png',dpi=170)
    plt.close(fig)

    mang=pd.read_csv(OUT/'manglish_profile.csv')
    mp=mang.pivot_table(index='language',columns='split',values='rows',aggfunc='sum',fill_value=0)
    fig,ax=plt.subplots(figsize=(9,4.8))
    mp.plot(kind='bar',ax=ax)
    ax.set_title('Manglish dataset language categories (source labels)')
    ax.set_ylabel('Records')
    ax.set_xlabel('')
    ax.tick_params(axis='x',rotation=0)
    fig.tight_layout()
    fig.savefig(OUT/'manglish_language_distribution.png',dpi=170)
    plt.close(fig)

    metadata=json.loads((OUT/'preparation_metadata.json').read_text(encoding='utf-8'))
    source=pd.read_csv(OUT/'source_language_label_distribution.csv')
    summary=[
      '# Phase 1 — Dataset Quality & EDA',
      '',
      '## Prepared splits',
      *[f'- **{k}**: {v:,} comments' for k,v in metadata['splits'].items()],
      '',
      '## Source and label cautions',
      '- MYBully `Hate` and bilingual `label` may follow different annotation policies. Do not present combined binary labels as fully harmonized.',
      '- MYBully original validation is kept separate and checked for duplicate overlap against training.',
      '- The bilingual dataset contains heterogeneous sources, including non-hate-only sources; review per-source evaluation before claiming robustness.',
      '- Manglish train/test have sentiment/emotion labels, not hate-speech labels; they were **not** used to train the hate classifier.',
      '- Source metadata and MYBully manual/HITL annotation types are kept in the output CSV.',
      '- Existing rows are deduplicated by NFKC/casefold/whitespace; more advanced near-duplicate review may still be needed.',
      '- Chinese and tri-language Rojak coverage is **not** established by these datasets.',
      '',
      '## Generated figures',
      '- `hate_class_distribution.png`',
      '- `manglish_language_distribution.png`',
      '',
      '## Label distribution',
      '```',table.to_string(index=False),'```',
      '',
      '## Bilingual source breakdown (all splits)',
      '```',source.loc[source.dataset!='MYBully'].groupby(['source_detail','hate_label'])['rows'].sum().to_string(),'```',
      '',
      '## Next step',
      'Train a character/word TF-IDF + Linear SVM binary baseline and report macro F1 separately for bilingual test and MYBully external validation. Select settings only on validation data.'
    ]
    (OUT/'EDA_REPORT.md').write_text('\n'.join(summary),encoding='utf-8')
    print('Saved charts and EDA_REPORT.md to',OUT)

if __name__=='__main__':
    main()
