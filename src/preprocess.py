"""
Data Preprocessing Module
- Cleans raw text (remove punctuation, lowercase, stopwords)
- Combines title + text
- TF-IDF vectorization
- Train/test split
- Saves vectorizer and processed arrays
"""

import pandas as pd
import numpy as np
import re
import string
import os
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data (run once)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()


# ── Text Cleaning ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Cleans a raw news article string.
    Steps: lowercase → remove URLs → remove punctuation → remove numbers
           → remove stopwords → stem words
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize and remove stopwords + short tokens
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # Stem tokens
    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


# ── Load & Prepare Data ──────────────────────────────────────
def load_data(fake_path: str, true_path: str) -> pd.DataFrame:
    """Loads raw CSVs, adds labels, combines and shuffles."""
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combines title + text, cleans, adds word count feature."""
    df = df.copy()

    # Combine title and text for richer features
    df["combined"] = df["title"].astype(str) + " " + df["text"].astype(str)

    print("🧹 Cleaning text... (this may take ~1-2 minutes)")
    df["cleaned"] = df["combined"].apply(clean_text)

    return df


# ── Vectorize ────────────────────────────────────────────────
def vectorize(df: pd.DataFrame, max_features: int = 50000):
    """
    Applies TF-IDF vectorization.
    Returns X (sparse matrix), y (labels), and the fitted vectorizer.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),       # unigrams + bigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True         # Apply log normalization
    )

    X = tfidf.fit_transform(df["cleaned"])
    y = df["label"].values

    return X, y, tfidf


# ── Train/Test Split ─────────────────────────────────────────
def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Splits data into training and test sets."""
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


# ── Main Preprocessing Pipeline ──────────────────────────────
def run_preprocessing():
    """Full preprocessing pipeline — call this script directly."""
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print(" Loading data...")
    df = load_data("data/raw/Fake.csv", "data/raw/True.csv")
    print(f"   Total samples: {len(df)}")

    print("  Preparing features...")
    df = prepare_features(df)

    print(" Vectorizing with TF-IDF...")
    X, y, tfidf = vectorize(df)
    print(f"   Feature matrix shape: {X.shape}")

    print("  Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"   Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # Save vectorizer and processed data
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    joblib.dump((X_train, X_test, y_train, y_test), "models/train_test_data.pkl")

    print("\n Preprocessing complete!")
    print("   Saved: models/tfidf_vectorizer.pkl")
    print("   Saved: models/train_test_data.pkl")

    return X_train, X_test, y_train, y_test, tfidf


if __name__ == "__main__":
    run_preprocessing()