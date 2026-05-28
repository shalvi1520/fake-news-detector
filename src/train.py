"""
Model Training Module
Trains 4 classifiers and saves them with performance metrics.
Models: Logistic Regression, Naive Bayes, Random Forest, Passive Aggressive
"""

import os
import json
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ── Model Definitions ────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42
    ),
    "Naive Bayes": MultinomialNB(
        alpha=0.1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=42
    ),
    "Passive Aggressive": PassiveAggressiveClassifier(
        max_iter=1000,
        random_state=42
    ),
}


# ── Training ─────────────────────────────────────────────────
def train_model(name: str, model, X_train, y_train):
    """Trains a single model and returns it."""
    print(f"   Training {name}...")
    model.fit(X_train, y_train)
    print(f"   {name} trained!")
    return model


def train_all_models(X_train, X_test, y_train, y_test):
    """
    Trains all models, evaluates them, saves each model,
    and returns a results dict.
    """
    os.makedirs("models", exist_ok=True)
    results = {}

    for name, model in MODELS.items():
        print(f"\n{'─'*40}")
        trained = train_model(name, model, X_train, y_train)

        # Evaluate
        y_pred = trained.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")

        results[name] = {
            "accuracy": round(acc * 100, 2),
            "f1_score": round(f1 * 100, 2),
        }

        print(f"   Accuracy : {acc*100:.2f}%")
        print(f"   F1 Score : {f1*100:.2f}%")

        # Save model
        safe_name = name.lower().replace(" ", "_")
        save_path = f"models/{safe_name}.pkl"
        joblib.dump(trained, save_path)
        print(f"   Saved: {save_path}")

    # Find best model
    best_model_name = max(results, key=lambda k: results[k]["accuracy"])
    results["best_model"] = best_model_name
    print(f"\n Best Model: {best_model_name} ({results[best_model_name]['accuracy']}% accuracy)")

    # Save results summary
    with open("models/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(" Saved: models/results.json")

    return results


# ── Main ─────────────────────────────────────────────────────
def run_training():
    """Load processed data and train all models."""
    print(" Loading preprocessed data...")
    X_train, X_test, y_train, y_test = joblib.load("models/train_test_data.pkl")
    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

    print("\n Starting model training...\n")
    results = train_all_models(X_train, X_test, y_train, y_test)

    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    for name, metrics in results.items():
        if name == "best_model":
            continue
        print(f"{name:<25} Acc: {metrics['accuracy']}%  F1: {metrics['f1_score']}%")

    return results


if __name__ == "__main__":
    run_training()