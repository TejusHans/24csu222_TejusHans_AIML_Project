from pathlib import Path
import argparse
import json
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent
MODELS = {'Original BERT': ROOT / 'trained_model_original', 'Improved BERT': ROOT / 'trained_model_experiment2'}
LABELS = ['FAKE', 'REAL']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='independent_examples.csv')
    args = parser.parse_args()
    path = ROOT / args.csv
    if not path.exists():
        raise SystemExit(f'Missing {path.name}. Fill independent_examples_template.csv, save as {path.name}, then retry.')
    df = pd.read_csv(path, keep_default_na=False)
    required = {'text', 'label', 'source_url'}
    if not required.issubset(df.columns):
        raise SystemExit(f'CSV must include columns: {", ".join(sorted(required))}')
    df['label'] = df['label'].astype(str).str.upper().str.strip()
    df['text'] = df['text'].astype(str).str.strip()
    df['source_url'] = df['source_url'].astype(str).str.strip()
    if df.empty or (df['text'].str.len() < 20).any() or not df['label'].isin(LABELS).all() or (df['source_url'] == '').any():
        raise SystemExit('Every row needs text >=20 characters, label FAKE/REAL, and a verification source_url.')
    if df['text'].str.lower().duplicated().any():
        raise SystemExit('Duplicate texts found; remove duplicates before evaluation.')
    if df['label'].nunique() < 2:
        raise SystemExit('Include verified examples from both FAKE and REAL classes.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = ROOT / 'results' / 'independent'
    output.mkdir(parents=True, exist_ok=True)
    scores = {}
    predictions = df.copy()
    for name, model_dir in MODELS.items():
        if not (model_dir / 'config.json').exists():
            raise SystemExit(f'Model missing: {model_dir}')
        print(f'Loading {name}...')
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True).to(device)
        model.eval()
        pred_labels, confidences = [], []
        for i, text in enumerate(df['text'], 1):
            inputs = tokenizer(text, truncation=True, max_length=128, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
            idx = int(probs.argmax().item())
            pred_labels.append(str(model.config.id2label[idx]).upper())
            confidences.append(round(float(probs[idx]) * 100, 2))
            print(f'{name}: {i}/{len(df)}', end='\r')
        print()
        key = 'original' if name.startswith('Original') else 'improved'
        predictions[f'{key}_prediction'] = pred_labels
        predictions[f'{key}_confidence'] = confidences
        report = classification_report(df['label'], pred_labels, labels=LABELS, output_dict=True, zero_division=0)
        matrix = confusion_matrix(df['label'], pred_labels, labels=LABELS)
        scores[key] = {'model': name, 'n': len(df), 'accuracy': float(accuracy_score(df['label'], pred_labels)), 'classification_report': report, 'confusion_matrix': matrix.tolist(), 'label_order': LABELS}
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'{name} — Independent Test')
        fig.tight_layout()
        fig.savefig(output / f'{key}_confusion_matrix.png', dpi=180)
        plt.close(fig)
    predictions.to_csv(output / 'predictions.csv', index=False)
    with open(output / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'note': 'User-provided verification labels; review source quality and dataset overlap before interpreting.', 'models': scores}, f, indent=2)
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    x = range(len(metrics))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, (key, info) in enumerate(scores.items()):
        vals = [info['accuracy']] + [info['classification_report']['macro avg'][m] for m in metrics[1:]]
        ax.bar([i + (j - 0.5) * 0.36 for i in x], vals, width=0.36, label=info['model'])
    ax.set_xticks(list(x), ['Accuracy', 'Precision', 'Recall', 'F1-score'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'model_comparison.png', dpi=180)
    plt.close(fig)
    print('\nSaved outputs to:', output)
    for info in scores.values():
        print(f"{info['model']}: accuracy = {info['accuracy']:.2%} (n={info['n']})")

if __name__ == '__main__':
    main()
