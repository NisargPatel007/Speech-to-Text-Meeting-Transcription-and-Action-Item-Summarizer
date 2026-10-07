"""
Training and Evaluation of Traditional Machine Learning Models:
1. Model 1: Logistic Regression (L2 regularized, class_weight='balanced')
2. Model 2: Linear Support Vector Classifier (LinearSVC, class_weight='balanced', calibrated)
Evaluates across:
- Validation Set
- Weak Test Set
- Human-Verified Test Set (200 samples)
Generates:
- Confusion Matrix Figures (fig7, fig8)
- Serialized Model Artifacts (.joblib)
- Results Metric CSVs
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from src.features import FeaturePipeline
from typing import Dict, Tuple, Any

PRIMARY_COLOR = "#2b5c8f"
ACCENT_COLOR = "#d95f02"


def evaluate_model(
    model: Any,
    X: Any,
    y_true: np.ndarray,
    model_name: str,
    dataset_name: str
) -> Dict[str, float]:
    """Computes comprehensive evaluation metrics."""
    y_pred = model.predict(X)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X)
    else:
        y_prob = y_pred

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1_pos = f1_score(y_true, y_pred, average="binary", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    metrics = {
        "Model": model_name,
        "Split": dataset_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Action_Item_F1": round(f1_pos, 4),
        "Macro_F1": round(f1_macro, 4),
        "ROC_AUC": round(auc, 4)
    }
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_png: str,
    fig_title: str
):
    """Generates and saves publication-quality Confusion Matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        ax=ax,
        xticklabels=["Non-Action (0)", "Action Item (1)"],
        yticklabels=["Non-Action (0)", "Action Item (1)"],
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title(fig_title, pad=14, fontsize=13, weight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, weight="semibold")
    ax.set_ylabel("True Label", fontsize=11, weight="semibold")

    # Add accuracy/percentages annotation inside cells
    total = cm.sum()
    pcts = cm.astype(float) / total * 100
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.72, f"({pcts[i, j]:.1f}%)", ha="center", va="center", color="#555555", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix to {output_png}")


def train_and_evaluate_traditional_models(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    figures_dir: str = "figures",
    results_dir: str = "results"
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Preprocessed Datasets
    print("[Traditional ML] Loading preprocessed datasets...")
    train_df = pd.read_csv(os.path.join(processed_dir, "train_preprocessed.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val_preprocessed.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test_preprocessed.csv"))
    verified_df = pd.read_csv(os.path.join(processed_dir, "manual_verification_preprocessed.csv"))

    # 2. Extract Features
    pipeline = FeaturePipeline(max_features=5000, ngram_range=(1, 2))
    X_train = pipeline.fit_transform(train_df)
    X_val = pipeline.transform(val_df)
    X_test = pipeline.transform(test_df)
    X_verified = pipeline.transform(verified_df)

    # Save fitted feature pipeline
    pipeline.save_pipeline(os.path.join(models_dir, "tfidf_vectorizer.joblib"))

    # Compute Chi-Square feature selection
    top_chi2 = pipeline.compute_chi2_feature_importance(train_df, top_k=20, results_dir=results_dir)

    y_train = train_df["weak_label"].values
    y_val = val_df["weak_label"].values
    y_test = test_df["weak_label"].values
    y_verified = verified_df["verified_label"].values  # Ground truth human verification label!

    all_metrics = []
    verified_metrics = []

    # -------------------------------------------------------------
    # MODEL 1: Logistic Regression
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Traditional ML] Training Model 1: Logistic Regression (Balanced)...")
    logreg = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    logreg.fit(X_train, y_train)

    joblib.dump(logreg, os.path.join(models_dir, "logistic_regression.joblib"))
    print("[Traditional ML] Saved models/logistic_regression.joblib")

    # Evaluate LogReg
    val_m_lr = evaluate_model(logreg, X_val, y_val, "Logistic Regression", "Validation")
    test_m_lr = evaluate_model(logreg, X_test, y_test, "Logistic Regression", "Weak Test")
    ver_m_lr = evaluate_model(logreg, X_verified, y_verified, "Logistic Regression", "Human-Verified Test")

    all_metrics.extend([val_m_lr, test_m_lr])
    verified_metrics.append(ver_m_lr)

    # Confusion Matrix for LogReg on Weak Test and Verified Test
    cm_lr_test = confusion_matrix(y_test, logreg.predict(X_test))
    plot_confusion_matrix(
        cm_lr_test,
        "Logistic Regression",
        os.path.join(figures_dir, "fig7_confusion_matrix_logreg.png"),
        "Figure 7: Confusion Matrix - Logistic Regression (Test Set)"
    )

    # -------------------------------------------------------------
    # MODEL 2: Linear SVM (Calibrated for Probabilities)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Traditional ML] Training Model 2: Linear Support Vector Classifier (LinearSVC)...")
    base_svm = LinearSVC(
        C=1.0,
        class_weight="balanced",
        loss="squared_hinge",
        random_state=42,
        max_iter=2000
    )
    # CalibratedClassifierCV allows valid probability outputs and smooth decision boundary
    svm = CalibratedClassifierCV(estimator=base_svm, cv=3)
    svm.fit(X_train, y_train)

    joblib.dump(svm, os.path.join(models_dir, "linear_svm.joblib"))
    print("[Traditional ML] Saved models/linear_svm.joblib")

    # Evaluate SVM
    val_m_svm = evaluate_model(svm, X_val, y_val, "Linear SVM", "Validation")
    test_m_svm = evaluate_model(svm, X_test, y_test, "Linear SVM", "Weak Test")
    ver_m_svm = evaluate_model(svm, X_verified, y_verified, "Linear SVM", "Human-Verified Test")

    all_metrics.extend([val_m_svm, test_m_svm])
    verified_metrics.append(ver_m_svm)

    # Confusion Matrix for SVM
    cm_svm_test = confusion_matrix(y_test, svm.predict(X_test))
    plot_confusion_matrix(
        cm_svm_test,
        "Linear SVM",
        os.path.join(figures_dir, "fig8_confusion_matrix_svm.png"),
        "Figure 8: Confusion Matrix - Linear Support Vector Machine (Test Set)"
    )

    # -------------------------------------------------------------
    # Save & Print Results
    # -------------------------------------------------------------
    df_metrics = pd.DataFrame(all_metrics)
    metrics_csv = os.path.join(results_dir, "traditional_models_metrics.csv")
    df_metrics.to_csv(metrics_csv, index=False)

    df_verified = pd.DataFrame(verified_metrics)
    verified_csv = os.path.join(results_dir, "traditional_models_verified_test_metrics.csv")
    df_verified.to_csv(verified_csv, index=False)

    print("\n" + "=" * 70)
    print("TRADITIONAL ML EVALUATION SUMMARY (Validation & Weak Test Splits):")
    print(df_metrics.to_string(index=False))

    print("\n" + "=" * 70)
    print("HONEST EVALUATION ON HUMAN-VERIFIED TEST SET (200 UTTERANCES):")
    print(df_verified.to_string(index=False))
    print("=" * 70)

    return df_metrics, df_verified


if __name__ == "__main__":
    train_and_evaluate_traditional_models()
