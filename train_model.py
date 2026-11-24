import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    fake_path = ROOT / "data" / "Fake.csv"
    true_path = ROOT / "data" / "True.csv"
    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError("Put Fake.csv and True.csv inside the data folder.")

    fake = pd.read_csv(fake_path)
    real = pd.read_csv(true_path)
    for df in (fake, real):
        if "text" not in df.columns:
            raise ValueError("Both CSV files must contain a text column")
    fake = fake.assign(label=0)
    real = real.assign(label=1)
    n = min(args.samples_per_class, len(fake), len(real))
    if n < 10:
        raise ValueError("Need at least 10 articles per class")
    df = pd.concat([fake.sample(n, random_state=42), real.sample(n, random_state=42)], ignore_index=True)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 20].drop_duplicates(subset=["text"])
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label={0: "FAKE", 1: "REAL"},
        label2id={"FAKE": 0, "REAL": 1}
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(NewsDataset(train_df.text, train_df.label, tokenizer), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(NewsDataset(test_df.text, test_df.label, tokenizer), batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | step {step}/{len(train_loader)} | loss {total_loss/step:.4f}", flush=True)
        print(f"Epoch {epoch+1} mean training loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels")
            outputs = model(**{k: v.to(device) for k, v in batch.items()})
            predictions.extend(outputs.logits.argmax(dim=1).cpu().tolist())
            truths.extend(labels.tolist())

    report = classification_report(truths, predictions, labels=[0, 1], target_names=["FAKE", "REAL"], output_dict=True, zero_division=0)
    matrix = confusion_matrix(truths, predictions, labels=[0, 1]).tolist()
    result = {"model": model_name, "device": str(device), "seed": 42, "train_size": len(train_df), "test_size": len(test_df), "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": 2e-5, "accuracy": accuracy_score(truths, predictions), "classification_report": report, "confusion_matrix": matrix, "label_order": ["FAKE", "REAL"]}
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    model_dir = ROOT / "trained_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("\nAccuracy:", result["accuracy"])
    print(classification_report(truths, predictions, labels=[0, 1], target_names=["FAKE", "REAL"], zero_division=0))
    print("Confusion matrix:", matrix)
    print("Saved model to:", model_dir)
    print("Saved metrics to:", results_dir / "metrics.json")

if __name__ == "__main__":
    main()
