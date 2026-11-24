import argparse, json, random, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts=list(texts); self.labels=list(labels); self.tokenizer=tokenizer; self.max_length=max_length
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        enc=self.tokenizer(self.texts[i], max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        item={k:v.squeeze(0) for k,v in enc.items()}
        item['labels']=torch.tensor(self.labels[i],dtype=torch.long)
        return item

def clean(text):
    text=str(text)
    # Remove leading wire-service dateline, e.g. WASHINGTON (Reuters) - ...
    text=re.sub(r'^\s*.{0,100}?\(Reuters\)\s*[-–—]\s*','',text,flags=re.I)
    text=re.sub(r'\bReuters\b',' ',text,flags=re.I)
    text=re.sub(r'https?://\S+|www\.\S+',' ',text)
    text=re.sub(r'\s+',' ',text).strip()
    return text

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--samples-per-class',type=int,default=500)
    ap.add_argument('--epochs',type=int,default=1)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--max-length',type=int,default=128)
    args=ap.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    parts=[]
    for filename,label in [('Fake.csv',0),('True.csv',1)]:
        data=pd.read_csv(ROOT/'data'/filename)
        if 'text' not in data: raise ValueError(f'{filename} missing text column')
        data=data[['text']].copy(); data['label']=label
        parts.append(data)
    df=pd.concat(parts,ignore_index=True)
    df['text']=df['text'].fillna('').map(clean)
    df=df[df['text'].str.len()>40].copy()
    df['normalized']=df['text'].str.lower().str.replace(r'[^a-z0-9]+',' ',regex=True).str.strip()
    # Remove conflicting-label duplicates rather than arbitrarily deciding truth.
    conflicts=df.groupby('normalized')['label'].nunique()
    conflicting=set(conflicts[conflicts>1].index)
    df=df[~df['normalized'].isin(conflicting)]
    df=df.drop_duplicates('normalized').drop(columns='normalized')
    # Sample after cleaning and de-duplication.
    n=min(args.samples_per_class,*df['label'].value_counts().reindex([0,1],fill_value=0).tolist())
    if n<10: raise ValueError('Too few clean examples per class')
    df=pd.concat([df[df.label==i].sample(n,random_state=42) for i in [0,1]],ignore_index=True)
    train,test=train_test_split(df,test_size=.2,random_state=42,stratify=df['label'])
    print(f'Cleaned sample: {len(df)} | Train: {len(train)} | Test: {len(test)}',flush=True)
    name='bert-base-uncased'
    tokenizer=AutoTokenizer.from_pretrained(name)
    model=AutoModelForSequenceClassification.from_pretrained(name,num_labels=2,id2label={0:'FAKE',1:'REAL'},label2id={'FAKE':0,'REAL':1})
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    train_loader=DataLoader(NewsDataset(train.text,train.label,tokenizer,args.max_length),batch_size=args.batch_size,shuffle=True)
    test_loader=DataLoader(NewsDataset(test.text,test.label,tokenizer,args.max_length),batch_size=args.batch_size)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
    for epoch in range(args.epochs):
        model.train(); total=0.
        for step,batch in enumerate(train_loader,1):
            batch={k:v.to(device) for k,v in batch.items()}
            optimizer.zero_grad(); loss=model(**batch).loss; loss.backward(); optimizer.step(); total+=loss.item()
            if step%10==0: print(f'Epoch {epoch+1}/{args.epochs}, step {step}/{len(train_loader)}, loss={total/step:.4f}',flush=True)
    model.eval(); pred=[]; truth=[]
    with torch.no_grad():
        for batch in test_loader:
            labels=batch.pop('labels'); logits=model(**{k:v.to(device) for k,v in batch.items()}).logits
            pred.extend(logits.argmax(1).cpu().tolist()); truth.extend(labels.tolist())
    report=classification_report(truth,pred,labels=[0,1],target_names=['FAKE','REAL'],output_dict=True,zero_division=0)
    metrics={'experiment':'cleaned-source-bias-check','model':name,'device':str(device),'seed':42,'train_size':len(train),'test_size':len(test),'epochs':args.epochs,'batch_size':args.batch_size,'learning_rate':2e-5,'accuracy':accuracy_score(truth,pred),'classification_report':report,'confusion_matrix':confusion_matrix(truth,pred,labels=[0,1]).tolist(),'label_order':['FAKE','REAL'],'limitations':'Reuters marker cleaning and exact normalized deduplication do not remove all publisher/style bias or near-duplicates.'}
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'metrics_experiment2.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    model_dir=ROOT/'trained_model_experiment2';model.save_pretrained(model_dir);tokenizer.save_pretrained(model_dir)
    print(json.dumps({'accuracy':metrics['accuracy'],'confusion_matrix':metrics['confusion_matrix'],'saved_to':str(out/'metrics_experiment2.json')},indent=2))
if __name__=='__main__':main()
