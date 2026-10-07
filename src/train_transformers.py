"""
Transformer Fine-Tuning Module for Action Item Classification.
Models:
1. Model 4: DistilBERT ('distilbert-base-uncased')
2. Model 5: RoBERTa ('roberta-base')

Hyperparameters per assignment rules:
- Hugging Face Trainer
- 3 Epochs
- Batch size 16 (or batch 8 with gradient_accumulation_steps 2 for 4GB VRAM safety)
- Learning rate 2e-5 (AdamW with linear warmup)
- Max length 128
- FP16 mixed precision
- Class-weighted Cross-Entropy loss for class imbalance
- Evaluated on Validation, Weak Test, and Human-Verified Benchmark
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

PRIMARY_COLOR = "#2b5c8f"
ACCENT_COLOR = "#d95f02"


def compute_metrics(eval_pred):
    """Computes comprehensive evaluation metrics for Trainer."""
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary", zero_division=0)
    rec = recall_score(labels, preds, average="binary", zero_division=0)
    f1_pos = f1_score(labels, preds, average="binary", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "action_item_f1": round(f1_pos, 4),
        "macro_f1": round(f1_macro, 4),
        "roc_auc": round(auc, 4)
    }


class WeightedTrainer(Trainer):
    """Custom Trainer subclass applying class weights to handle imbalance."""
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def plot_transformer_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_png: str
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
    ax.set_title(f"Figure: Confusion Matrix - {model_name} (Test Set)", pad=14, weight="bold")
    ax.set_xlabel("Predicted Label", weight="semibold")
    ax.set_ylabel("True Label", weight="semibold")

    total = cm.sum()
    pcts = cm.astype(float) / total * 100
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.72, f"({pcts[i, j]:.1f}%)", ha="center", va="center", color="#555555", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[Transformer] Saved confusion matrix to {output_png}")


def fine_tune_transformer_model(
    model_name_or_path: str,
    output_dir: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    verified_df: pd.DataFrame,
    epochs: int = 3,
    batch_size: int = 8,
    grad_accum: int = 2,
    lr: float = 2e-5,
    max_len: int = 128,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray]:
    """
    Fine-tunes a Transformer model using Hugging Face Trainer with FP16 precision.
    """
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Transformer] Initializing {model_name_or_path} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2).to(device)

    # Prepare HF Datasets
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding=False, truncation=True, max_length=max_len)

    train_ds = Dataset.from_dict({"text": train_df["cleaned_text"].fillna("").tolist(), "labels": train_df["weak_label"].tolist()})
    val_ds = Dataset.from_dict({"text": val_df["cleaned_text"].fillna("").tolist(), "labels": val_df["weak_label"].tolist()})
    test_ds = Dataset.from_dict({"text": test_df["cleaned_text"].fillna("").tolist(), "labels": test_df["weak_label"].tolist()})
    ver_ds = Dataset.from_dict({"text": verified_df["cleaned_text"].fillna("").tolist(), "labels": verified_df["verified_label"].tolist()})

    train_tokenized = train_ds.map(tokenize_fn, batched=True)
    val_tokenized = val_ds.map(tokenize_fn, batched=True)
    test_tokenized = test_ds.map(tokenize_fn, batched=True)
    ver_tokenized = ver_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=16,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_total_limit=1,
        report_to="none"
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    print(f"[Transformer] Starting fine-tuning for {epochs} epochs...")
    train_result = trainer.train()

    # Save final best model & tokenizer
    final_save_path = output_dir
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"[Transformer] Model saved to {final_save_path}")

    # Evaluate on splits
    def eval_hf(split_ds, split_name):
        res = trainer.predict(split_ds)
        metrics = res.metrics
        logits = res.predictions
        preds = np.argmax(logits, axis=-1)
        cm = confusion_matrix(res.label_ids, preds)
        formatted = {
            "Model": os.path.basename(output_dir).replace("_action_item", "").capitalize(),
            "Split": split_name,
            "Accuracy": metrics["test_accuracy"],
            "Precision": metrics["test_precision"],
            "Recall": metrics["test_recall"],
            "Action_Item_F1": metrics["test_action_item_f1"],
            "Macro_F1": metrics["test_macro_f1"],
            "ROC_AUC": metrics["test_roc_auc"]
        }
        return formatted, cm

    val_res, _ = eval_hf(val_tokenized, "Validation")
    test_res, test_cm = eval_hf(test_tokenized, "Weak Test")
    ver_res, _ = eval_hf(ver_tokenized, "Human-Verified Test")

    # Cleanup temporary checkpoints to conserve disk space
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    return train_result, [val_res, test_res, ver_res], test_cm


def run_transformers_pipeline(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    figures_dir: str = "figures",
    results_dir: str = "results"
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Data
    train_df = pd.read_csv(os.path.join(processed_dir, "train_preprocessed.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val_preprocessed.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test_preprocessed.csv"))
    verified_df = pd.read_csv(os.path.join(processed_dir, "manual_verification_preprocessed.csv"))

    # Compute Class Weights: [weight_neg, weight_pos]
    n_neg = (train_df["weak_label"] == 0).sum()
    n_pos = (train_df["weak_label"] == 1).sum()
    total = len(train_df)
    w0 = total / (2.0 * n_neg)
    w1 = total / (2.0 * n_pos)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32)
    print(f"[Transformer] Class weights for Cross-Entropy: Class 0 = {w0:.3f}, Class 1 = {w1:.3f}")

    all_transformer_metrics = []

    # -------------------------------------------------------------
    # MODEL 4: DistilBERT ('distilbert-base-uncased')
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL 4: DISTILBERT FINE-TUNING ('distilbert-base-uncased')")
    print("=" * 70)
    distilbert_dir = os.path.join(models_dir, "distilbert_action_item")
    _, distil_metrics, distil_cm = fine_tune_transformer_model(
        model_name_or_path="distilbert-base-uncased",
        output_dir=distilbert_dir,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        verified_df=verified_df,
        epochs=3,
        batch_size=8,
        grad_accum=2,
        lr=2e-5,
        max_len=128,
        class_weights=class_weights
    )
    all_transformer_metrics.extend(distil_metrics)

    # Save DistilBERT confusion matrix
    plot_transformer_confusion_matrix(
        distil_cm,
        "DistilBERT",
        os.path.join(figures_dir, "fig12_confusion_matrix_distilbert.png")
    )

    # -------------------------------------------------------------
    # MODEL 5: RoBERTa ('roberta-base')
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL 5: ROBERTA FINE-TUNING ('roberta-base')")
    print("=" * 70)
    roberta_dir = os.path.join(models_dir, "roberta_action_item")
    _, roberta_metrics, roberta_cm = fine_tune_transformer_model(
        model_name_or_path="roberta-base",
        output_dir=roberta_dir,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        verified_df=verified_df,
        epochs=3,
        batch_size=8,
        grad_accum=2,
        lr=2e-5,
        max_len=128,
        class_weights=class_weights
    )
    all_transformer_metrics.extend(roberta_metrics)

    # Save RoBERTa confusion matrix
    plot_transformer_confusion_matrix(
        roberta_cm,
        "RoBERTa",
        os.path.join(figures_dir, "fig13_confusion_matrix_roberta.png")
    )

    # Save Transformer Results
    df_trans_metrics = pd.DataFrame(all_transformer_metrics)
    trans_csv = os.path.join(results_dir, "transformer_models_metrics.csv")
    df_trans_metrics.to_csv(trans_csv, index=False)

    # -------------------------------------------------------------
    # MASTER COMPARISON TABLE ACROSS ALL 5 MODELS
    # -------------------------------------------------------------
    # Load Traditional and BiLSTM metrics
    trad_df = pd.read_csv(os.path.join(results_dir, "traditional_models_metrics.csv"))
    trad_ver_df = pd.read_csv(os.path.join(results_dir, "traditional_models_verified_test_metrics.csv"))
    bilstm_df = pd.read_csv(os.path.join(results_dir, "bilstm_metrics.csv"))

    all_dfs = [trad_df, trad_ver_df, bilstm_df, df_trans_metrics]
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_csv = os.path.join(results_dir, "all_5_models_master_comparison.csv")
    master_df.to_csv(master_csv, index=False)

    # Plot Fig 14: Comprehensive Model Comparison Bar Chart on Human-Verified Test Set
    ver_comparison = master_df[master_df["Split"] == "Human-Verified Test"].copy()
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    x = np.arange(len(ver_comparison))
    width = 0.25

    rects1 = ax.bar(x - width, ver_comparison["Accuracy"], width, label="Accuracy", color=PRIMARY_COLOR, alpha=0.85)
    rects2 = ax.bar(x, ver_comparison["Macro_F1"], width, label="Macro-F1", color=ACCENT_COLOR, alpha=0.85)
    rects3 = ax.bar(x + width, ver_comparison["Action_Item_F1"], width, label="Action-Item F1", color="#2ca02c", alpha=0.85)

    ax.set_ylabel("Score (0.0 - 1.0)", weight="semibold")
    ax.set_title("Figure 14: Master Comparison of All 5 Models on Human-Verified Ground Truth", pad=15, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ver_comparison["Model"], weight="semibold")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=True, loc="lower right")

    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            val = bar.get_height()
            ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, rotation=45)

    plt.tight_layout()
    fig14_path = os.path.join(figures_dir, "fig14_all_5_models_comparison.png")
    plt.savefig(fig14_path, dpi=300)
    plt.close()
    print(f"[Transformer] Saved {fig14_path}")

    print("\n" + "=" * 75)
    print("MASTER EVALUATION BENCHMARK: HUMAN-VERIFIED GROUND TRUTH (ALL 5 MODELS):")
    print(ver_comparison[["Model", "Accuracy", "Precision", "Recall", "Action_Item_F1", "Macro_F1", "ROC_AUC"]].to_string(index=False))
    print("=" * 75)

    return master_df


if __name__ == "__main__":
    run_transformers_pipeline()
