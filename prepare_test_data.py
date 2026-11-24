from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

# LIAR dataset: column 1 = ID, 2 = label, 3 = statement
df = pd.read_csv(
    ROOT / "test.tsv",
    sep="\t",
    header=None,
    usecols=[0, 1, 2],
    names=["id", "original_label", "text"],
    dtype=str,
    quoting=3,
    on_bad_lines="skip",
)

# Use only clearly true or false labels
df = df[df["original_label"].isin(["true", "false"])].copy()

df["text"] = df["text"].fillna("").str.strip()
df = df[df["text"].str.len() >= 20]
df = df.drop_duplicates(subset=["text"])

real = df[df["original_label"] == "true"].sample(
    n=25, random_state=42
)

fake = df[df["original_label"] == "false"].sample(
    n=25, random_state=42
)

selected = pd.concat([real, fake], ignore_index=True)
selected = selected.sample(frac=1, random_state=42)

selected["label"] = selected["original_label"].map({
    "true": "REAL",
    "false": "FAKE"
})

selected["source_url"] = (
    "https://www.politifact.com/"
)

selected[["text", "label", "source_url"]].to_csv(
    ROOT / "independent_examples.csv",
    index=False,
    encoding="utf-8"
)

print("Independent dataset created successfully!")
print("Total examples:", len(selected))
print(selected["label"].value_counts())