"""
Evaluation Module
Generates:
- Confusion matrices for all models
- Classification reports
- ROC curves
- Model comparison bar chart
"""

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score
)

os.makedirs("assets", exist_ok=True)

MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "Passive Aggressive": "models/passive_aggressive.pkl",
}

LABEL_NAMES = ["Fake", "Real"]


# ── Confusion Matrix ──────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, model_name: str):
    """Saves a confusion matrix heatmap for one model."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                linewidths=0.5)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    path = f"assets/cm_{safe_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  💾 Saved: {path}")
    return cm


# ── ROC Curve ────────────────────────────────────────────────
def plot_roc_curves(models_data: dict):
    """Plots ROC curves for all models on a single chart."""
    plt.figure(figsize=(8, 6))

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    for (name, (y_test, y_proba)), color in zip(models_data.items(), colors):
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("assets/roc_curves.png", dpi=150)
    plt.close()
    print("  💾 Saved: assets/roc_curves.png")


# ── Model Comparison Bar Chart ────────────────────────────────
def plot_model_comparison(results: dict):
    """Bar chart comparing accuracy and F1 across all models."""
    names = [k for k in results if k != "best_model"]
    accuracies = [results[k]["accuracy"] for k in names]
    f1_scores = [results[k]["f1_score"] for k in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy %",
                   color="#3498db", edgecolor="white")
    bars2 = ax.bar(x + width/2, f1_scores, width, label="F1 Score %",
                   color="#2ecc71", edgecolor="white")

    ax.set_ylabel("Score (%)")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(85, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("assets/model_comparison.png", dpi=150)
    plt.close()
    print("  💾 Saved: assets/model_comparison.png")


# ── Main Evaluation Pipeline ──────────────────────────────────
def run_evaluation():
    print(" Loading test data and models...")
    X_train, X_test, y_train, y_test = joblib.load("models/train_test_data.pkl")

    with open("models/results.json") as f:
        results = json.load(f)

    roc_data = {}

    for name, path in MODEL_FILES.items():
        print(f"\n{'─'*40}")
        print(f" Evaluating: {name}")

        model = joblib.load(path)
        y_pred = model.predict(X_test)

        # Classification Report
        print(classification_report(y_test, y_pred,
                                    target_names=LABEL_NAMES))

        # Confusion Matrix
        plot_confusion_matrix(y_test, y_pred, name)

        # ROC data (if model supports predict_proba)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_data[name] = (y_test, y_proba)
        except AttributeError:
            roc_data[name] = (y_test, None)

    # ROC Curves
    print("\n Plotting ROC Curves...")
    plot_roc_curves(roc_data)

    # Model Comparison
    print(" Plotting Model Comparison...")
    plot_model_comparison(results)

    print("\n Evaluation complete! All plots saved in assets/")


if __name__ == "__main__":
    run_evaluation()