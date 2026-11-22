"""Optional controlled OCR robustness test on a small validation subset.

Constructs screenshots locally from ORIGINAL labelled validation comments; compares
classification from direct text versus OCR to isolate OCR errors. Requires EasyOCR,
Pillow, and a readable TrueType font. Never uses test labels to tune parameters.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from .common import load_splits,verify_baseline_config,load_saved_model,score_model,read_thresholds,metrics


def wrapped(text,width=65):
    import textwrap
    return '\n'.join(textwrap.wrap(text,width=width,break_long_words=True)[:8])


def rasterize(text,font_path,font_size=24,blur=0):
    font=ImageFont.truetype(font_path,font_size)
    img=Image.new('RGB',(1000,330),'white');draw=ImageDraw.Draw(img)
    draw.multiline_text((22,16),wrapped(text),font=font,fill='black',spacing=5)
    if blur:img=img.filter(ImageFilter.GaussianBlur(blur))
    return img


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='train.csv');p.add_argument('--sample',type=int,default=0)
    p.add_argument('--project-root',default='.')
    p.add_argument('--model',choices=['svm','xgboost'],default='xgboost')
    p.add_argument('--font',required=True,help='Path to existing TTF font, e.g. C:/Windows/Fonts/arial.ttf')
    p.add_argument('--count',type=int,default=50)
    p.add_argument('--out',default='part_b/results')
    a=p.parse_args()
    if a.count<1:raise ValueError('count must be >= 1')
    import easyocr
    root=Path(a.project_root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    texts,y,splits=load_splits(a.csv,a.sample)
    verify_baseline_config(root,a.csv,a.sample,len(texts))
    ix=splits['validation'][:a.count]
    original=[str(x) for x in texts[ix]]
    model=load_saved_model(root/'results'/'models'/f'{a.model}.joblib')
    thresholds=read_thresholds(root/'results'/f'{a.model}_thresholds.csv')
    reader=easyocr.Reader(['en'],gpu=False)
    rows=[]
    for setting,blur in [('clear',0),('blurred',1.8)]:
        extracted=[]
        for t in original:
            im=rasterize(t,a.font,blur=blur)
            extracted.append(' '.join(reader.readtext(np.asarray(im),detail=0,paragraph=True)))
        orig_values=score_model(model,a.model,original)
        ocr_values=score_model(model,a.model,extracted)
        base=metrics(y[ix],orig_values,thresholds)
        ocr=metrics(y[ix],ocr_values,thresholds)
        agreement=float(np.mean(np.all((orig_values>=thresholds)==(ocr_values>=thresholds),axis=1)))
        rows.append({'condition':setting,'comments':len(ix),
                     'direct_macro_f1':base['macro_f1'],'ocr_macro_f1':ocr['macro_f1'],
                     'direct_micro_f1':base['micro_f1'],'ocr_micro_f1':ocr['micro_f1'],
                     'exact_label_vector_agreement':agreement})
        pd.DataFrame({'original_text':original,'ocr_text':extracted}).to_csv(
            out/f'ocr_{setting}_paired_texts.csv',index=False)
        print(setting,rows[-1])
    pd.DataFrame(rows).to_csv(out/'ocr_robustness_validation.csv',index=False)

if __name__=='__main__':main()
