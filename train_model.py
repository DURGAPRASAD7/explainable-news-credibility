import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_welfake(path):
    df = pd.read_csv(path, sep=",")
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"title", "text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"WELFake needs columns {required}. Found: {list(df.columns)}")
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = (df["title"] + " " + df["text"]).map(clean_text)
    # WELFake: 0=fake, 1=real. Model target: 0=fake, 1=real.
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])]
    return df[["content", "label"]]

def load_isot(fake_path, true_path):
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    for df in (fake, true):
        df.columns = [str(c).strip().lower() for c in df.columns]
    fake["content"] = (fake.get("title", "").fillna("") + " " + fake.get("text", "").fillna("")).map(clean_text)
    true["content"] = (true.get("title", "").fillna("") + " " + true.get("text", "").fillna("")).map(clean_text)
    fake = fake[["content"]].assign(label=0)
    true = true[["content"]].assign(label=1)
    return pd.concat([fake, true], ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["welfake", "isot"], default="welfake")
    args = parser.parse_args()

    if args.dataset == "welfake":
        path = DATA / "WELFake_Dataset.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Put WELFake_Dataset.csv in {DATA}. "
                "See README.txt for the source."
            )
        df = load_welfake(path)
    else:
        fake_path, true_path = DATA / "Fake.csv", DATA / "True.csv"
        if not fake_path.exists() or not true_path.exists():
            raise FileNotFoundError("For ISOT, put Fake.csv and True.csv in data/")
        df = load_isot(fake_path, true_path)

    df = df.dropna(subset=["content", "label"])
    df["content"] = df["content"].astype(str)
    df = df[df["content"].str.len() >= 30]
    df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)

    print(f"Rows after cleaning: {len(df):,}")
    print("Class distribution:", df["label"].value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=120000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9'-]{1,}\b"
    )

    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1200,
        class_weight="balanced",
        C=2.0,
        solver="liblinear",
        random_state=42
    )
    model.fit(Xtr, y_train)

    pred = model.predict(Xte)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred).tolist()
    report = classification_report(
        y_test, pred, target_names=["Fake", "Real"], output_dict=True
    )

    joblib.dump(vectorizer, MODELS / "tfidf_vectorizer.joblib")
    joblib.dump(model, MODELS / "news_classifier.joblib")

    metadata = {
        "dataset": args.dataset,
        "rows": int(len(df)),
        "accuracy": round(float(acc), 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "label_mapping": {"0": "Fake", "1": "Real"},
        "model": "TF-IDF + Logistic Regression",
        "ngram_range": [1, 2]
    }
    (MODELS / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nTraining complete.")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, pred, target_names=["Fake", "Real"]))

if __name__ == "__main__":
    main()
