import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import PROCESSED_DIR, MODELS_DIR


def train():
    data_path = os.path.join(PROCESSED_DIR, "cfpb_clean.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found: {data_path}. Run preprocess first.")

    df = pd.read_csv(data_path)

    # Basic cleanup
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    if df["label"].nunique() < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2
        )),
        ("lr", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, zero_division=0))

    os.makedirs(MODELS_DIR, exist_ok=True)

    out_path = os.path.join(MODELS_DIR, "triage_model.joblib")
    joblib.dump(clf, out_path)
    print("Saved model:", out_path)

    # Save label list (nice for UI/README)
    labels_path = os.path.join(MODELS_DIR, "labels.txt")
    labels = sorted(df["label"].unique().tolist())
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))
    print("Saved labels:", labels_path)


if __name__ == "__main__":
    train()
