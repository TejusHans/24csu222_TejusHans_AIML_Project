import pandas as pd

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

for name, df in [("FAKE", fake), ("REAL", real)]:
    texts = df["text"].fillna("").astype(str)

    print(f"\n{name} DATASET")
    print("Total articles:", len(df))
    print("Reuters mentions:", texts.str.contains(
        "Reuters", case=False, regex=False
    ).sum())
    print("Duplicate articles:", texts.duplicated().sum())
    print("Average text length:", round(texts.str.len().mean()))