"""
Bidirectional LSTM (BiLSTM) Classifier for Utterance-Level Action Item Detection.
Architecture:
  Input Tokens -> Embedding(vocab_size, 128) -> BiLSTM(hidden_size=64, bidirectional=True)
  -> Dropout(0.3) -> Dense(128, 32) -> ReLU -> Dropout(0.2) -> Dense(32, 1) -> Sigmoid
Features:
- PyTorch implementation with CUDA GPU acceleration
- Weighted Binary Cross-Entropy Loss to handle class imbalance
- Early stopping on Validation Macro-F1
- Saves learning curves: fig9_bilstm_loss_curve.png, fig10_bilstm_f1_accuracy_curve.png
- Evaluates on Validation, Weak Test, and Human-Verified Benchmark
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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


# -------------------------------------------------------------
# 1. Dataset & Vocabulary Helpers
# -------------------------------------------------------------
class Vocabulary:
    def __init__(self, min_freq: int = 2, max_size: int = 10000):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
        self.idx2word = {self.pad_idx: self.pad_token, self.unk_idx: self.unk_token}

    def build_vocab(self, texts: List[str]):
        counter = Counter()
        for t in texts:
            tokens = str(t).split()
            counter.update(tokens)

        sorted_words = [w for w, c in counter.most_common() if c >= self.min_freq]
        if self.max_size:
            sorted_words = sorted_words[: self.max_size - 2]

        for w in sorted_words:
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w

    def numericalize(self, text: str, max_len: int = 64) -> List[int]:
        tokens = str(text).split()
        indices = [self.word2idx.get(tok, self.unk_idx) for tok in tokens[:max_len]]
        # Pad if needed
        if len(indices) < max_len:
            indices = indices + [self.pad_idx] * (max_len - len(indices))
        return indices

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "word2idx": self.word2idx,
                "min_freq": self.min_freq,
                "max_size": self.max_size
            }, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(min_freq=data["min_freq"], max_size=data["max_size"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in data["word2idx"].items()}
        return vocab


class UtteranceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, text_col: str = "processed_text", label_col: str = "weak_label", max_len: int = 64):
        self.labels = df[label_col].values.astype(np.float32) if label_col in df.columns else np.zeros(len(df), dtype=np.float32)
        self.sequences = [vocab.numericalize(t, max_len=max_len) for t in df[text_col].fillna("")]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.sequences[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


# -------------------------------------------------------------
# 2. PyTorch BiLSTM Neural Architecture
# -------------------------------------------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        dense_dim: int = 32,
        dropout_lstm: float = 0.3,
        dropout_dense: float = 0.2,
        pad_idx: int = 0
    ):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout_lstm)
        self.fc1 = nn.Linear(hidden_dim * 2, dense_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_dense)
        self.out_proj = nn.Linear(dense_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embed: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(input_ids)
        
        # LSTM output: [batch_size, seq_len, hidden_dim * 2]
        lstm_out, (hn, cn) = self.lstm(embedded)
        
        # Concatenate final forward and backward hidden states
        # hn shape: [num_layers * 2, batch_size, hidden_dim]
        h_forward = hn[-2, :, :]
        h_backward = hn[-1, :, :]
        hidden_cat = torch.cat([h_forward, h_backward], dim=1)  # [batch_size, hidden_dim * 2]

        dropped = self.dropout1(hidden_cat)
        dense1 = self.relu(self.fc1(dropped))
        dropped2 = self.dropout2(dense1)
        logits = self.out_proj(dropped2).squeeze(-1)  # [batch_size]
        return logits


# -------------------------------------------------------------
# 3. Training & Evaluation Engine
# -------------------------------------------------------------
def train_bilstm_model(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    figures_dir: str = "figures",
    results_dir: str = "results",
    batch_size: int = 64,
    epochs: int = 15,
    lr: float = 1e-3,
    max_len: int = 64,
    patience: int = 4
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BiLSTM] Training using device: {device}")
    if torch.cuda.is_available():
        print(f"[BiLSTM] GPU detected: {torch.cuda.get_device_name(0)}")

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load Data
    train_df = pd.read_csv(os.path.join(processed_dir, "train_preprocessed.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val_preprocessed.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test_preprocessed.csv"))
    verified_df = pd.read_csv(os.path.join(processed_dir, "manual_verification_preprocessed.csv"))

    # 2. Build Vocabulary
    vocab = Vocabulary(min_freq=2, max_size=10000)
    vocab.build_vocab(train_df["processed_text"].dropna().tolist())
    vocab_path = os.path.join(models_dir, "bilstm_vocab.json")
    vocab.save(vocab_path)
    print(f"[BiLSTM] Vocabulary size: {len(vocab.word2idx)} tokens. Saved to {vocab_path}")

    # 3. Create DataLoaders
    train_ds = UtteranceDataset(train_df, vocab, label_col="weak_label", max_len=max_len)
    val_ds = UtteranceDataset(val_df, vocab, label_col="weak_label", max_len=max_len)
    test_ds = UtteranceDataset(test_df, vocab, label_col="weak_label", max_len=max_len)
    verified_ds = UtteranceDataset(verified_df, vocab, label_col="verified_label", max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    verified_loader = DataLoader(verified_ds, batch_size=batch_size, shuffle=False)

    # 4. Handle Class Imbalance with pos_weight
    n_neg = (train_df["weak_label"] == 0).sum()
    n_pos = (train_df["weak_label"] == 1).sum()
    pos_weight_val = float(n_neg / max(n_pos, 1))
    pos_weight = torch.tensor([pos_weight_val], device=device)
    print(f"[BiLSTM] Class imbalance ratio: {pos_weight_val:.2f} (pos_weight applied to BCEWithLogitsLoss)")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 5. Model, Optimizer, Scheduler
    model = BiLSTMClassifier(
        vocab_size=len(vocab.word2idx),
        embedding_dim=128,
        hidden_dim=64,
        dense_dim=32,
        pad_idx=vocab.pad_idx
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Training loop with early stopping
    best_val_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history = []
    best_weights_path = os.path.join(models_dir, "bilstm_classifier.pt")

    print("\n[BiLSTM] Starting training loop...")
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_preds, train_trues = [], []

        for batch in train_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds.extend((probs >= 0.5).astype(int))
            train_trues.extend(labels.cpu().numpy().astype(int))

        train_loss = train_loss / len(train_ds)
        train_f1 = f1_score(train_trues, train_preds, average="macro", zero_division=0)
        train_acc = accuracy_score(train_trues, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_trues, val_probs = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_ids"].to(device)
                labels = batch["label"].to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)

                val_loss += loss.item() * len(labels)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_preds.extend((probs >= 0.5).astype(int))
                val_trues.extend(labels.cpu().numpy().astype(int))

        val_loss = val_loss / len(val_ds)
        val_f1 = f1_score(val_trues, val_preds, average="macro", zero_division=0)
        val_pos_f1 = f1_score(val_trues, val_preds, average="binary", zero_division=0)
        val_acc = accuracy_score(val_trues, val_preds)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "train_f1": round(train_f1, 4),
            "val_f1": round(val_f1, 4),
            "val_pos_f1": round(val_pos_f1, 4)
        })

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
        )

        # Early Stopping Check on Macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_weights_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[BiLSTM] Early stopping triggered at epoch {epoch}. Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}.")
                break

    # Save History
    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(results_dir, "bilstm_training_history.csv"), index=False)

    # 6. Plot Learning Curves
    # Fig 9: Loss Curve
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(df_history["epoch"], df_history["train_loss"], marker="o", color=PRIMARY_COLOR, label="Training Loss", linewidth=2.2)
    ax.plot(df_history["epoch"], df_history["val_loss"], marker="s", color=ACCENT_COLOR, label="Validation Loss", linewidth=2.2)
    ax.axvline(best_epoch, color="#2ca02c", linestyle="--", label=f"Best Model (Epoch {best_epoch})")
    ax.set_title("Figure 9: BiLSTM Binary Cross-Entropy Loss Curve", pad=14, weight="bold")
    ax.set_xlabel("Epoch", weight="semibold")
    ax.set_ylabel("Loss", weight="semibold")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig9_path = os.path.join(figures_dir, "fig9_bilstm_loss_curve.png")
    plt.savefig(fig9_path, dpi=300)
    plt.close()
    print(f"[BiLSTM] Saved {fig9_path}")

    # Fig 10: Accuracy & Macro-F1 Curve
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(df_history["epoch"], df_history["val_acc"], marker="o", color=PRIMARY_COLOR, label="Validation Accuracy", linewidth=2.2)
    ax.plot(df_history["epoch"], df_history["val_f1"], marker="^", color=ACCENT_COLOR, label="Validation Macro-F1", linewidth=2.2)
    ax.plot(df_history["epoch"], df_history["val_pos_f1"], marker="d", color="#7570b3", label="Action Item Class F1", linewidth=2.0, linestyle=":")
    ax.axvline(best_epoch, color="#2ca02c", linestyle="--", label=f"Early Stop Checkpoint (Epoch {best_epoch})")
    ax.set_title("Figure 10: BiLSTM Validation Accuracy and Macro-F1 Trajectory", pad=14, weight="bold")
    ax.set_xlabel("Epoch", weight="semibold")
    ax.set_ylabel("Score", weight="semibold")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig10_path = os.path.join(figures_dir, "fig10_bilstm_f1_accuracy_curve.png")
    plt.savefig(fig10_path, dpi=300)
    plt.close()
    print(f"[BiLSTM] Saved {fig10_path}")

    # 7. Final Model Evaluation on Best Checkpoint
    print(f"\n[BiLSTM] Reloading best checkpoint from {best_weights_path}...")
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()

    def eval_split(loader, split_name):
        preds, trues, probs = [], [], []
        with torch.no_grad():
            for batch in loader:
                inputs = batch["input_ids"].to(device)
                labels = batch["label"].to(device)
                logits = model(inputs)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.extend(p)
                preds.extend((p >= 0.5).astype(int))
                trues.extend(labels.cpu().numpy().astype(int))

        acc = accuracy_score(trues, preds)
        prec = precision_score(trues, preds, average="binary", zero_division=0)
        rec = recall_score(trues, preds, average="binary", zero_division=0)
        f1_pos = f1_score(trues, preds, average="binary", zero_division=0)
        f1_macro = f1_score(trues, preds, average="macro", zero_division=0)
        auc = roc_auc_score(trues, probs)

        cm = confusion_matrix(trues, preds)

        return {
            "Model": "BiLSTM",
            "Split": split_name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "Action_Item_F1": round(f1_pos, 4),
            "Macro_F1": round(f1_macro, 4),
            "ROC_AUC": round(auc, 4)
        }, cm

    val_metrics, _ = eval_split(val_loader, "Validation")
    test_metrics, test_cm = eval_split(test_loader, "Weak Test")
    ver_metrics, _ = eval_split(verified_loader, "Human-Verified Test")

    # Fig 11: Confusion Matrix for BiLSTM on Weak Test
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
    sns.heatmap(
        test_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        ax=ax,
        xticklabels=["Non-Action (0)", "Action Item (1)"],
        yticklabels=["Non-Action (0)", "Action Item (1)"],
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title("Figure 11: Confusion Matrix - BiLSTM Classifier (Test Set)", pad=14, weight="bold")
    ax.set_xlabel("Predicted Label", weight="semibold")
    ax.set_ylabel("True Label", weight="semibold")
    plt.tight_layout()
    fig11_path = os.path.join(figures_dir, "fig11_confusion_matrix_bilstm.png")
    plt.savefig(fig11_path, dpi=300)
    plt.close()
    print(f"[BiLSTM] Saved {fig11_path}")

    # Save metrics
    df_bilstm_metrics = pd.DataFrame([val_metrics, test_metrics, ver_metrics])
    metrics_path = os.path.join(results_dir, "bilstm_metrics.csv")
    df_bilstm_metrics.to_csv(metrics_path, index=False)

    print("\n" + "=" * 70)
    print("BILSTM EVALUATION SUMMARY:")
    print(df_bilstm_metrics.to_string(index=False))
    print("=" * 70)

    return df_bilstm_metrics


if __name__ == "__main__":
    train_bilstm_model()
