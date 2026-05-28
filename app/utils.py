"""
Utility functions for the Streamlit app.
"""

import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    """Same cleaning function used during training."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


def predict_news(text: str, model, vectorizer) -> dict:
    """
    Takes raw text, returns prediction dict with label and confidence.
    """
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    label = "REAL ✅" if pred == 1 else "FAKE ❌"

    try:
        proba = model.predict_proba(X)[0]
        confidence = max(proba) * 100
    except AttributeError:
        # Passive Aggressive has decision_function instead
        confidence = None

    return {
        "label": label,
        "is_real": bool(pred == 1),
        "confidence": confidence,
        "cleaned_text": cleaned,
    }


def load_models_and_vectorizer():
    """Loads all saved models and TF-IDF vectorizer."""
    import os
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    model_files = {
        "Logistic Regression": "models/logistic_regression.pkl",
        "Naive Bayes": "models/naive_bayes.pkl",
        "Random Forest": "models/random_forest.pkl",
        "Passive Aggressive": "models/passive_aggressive.pkl",
    }

    models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)

    return models, vectorizer

def fetch_article_from_url(url: str) -> dict:
    """
    Scrapes a news article from a URL using newspaper3k.
    Returns title, text, and publish date.
    """
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        return {
            "success": True,
            "title": article.title,
            "text": article.text,
            "publish_date": str(article.publish_date) if article.publish_date else "Unknown",
            "authors": article.authors,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }