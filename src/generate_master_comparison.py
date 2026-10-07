"""
Master Aggregation Script for all 5 Models.
Combines:
- Traditional ML: Logistic Regression, Linear SVM
- Deep Learning: BiLSTM
- Transformers: DistilBERT, RoBERTa
Generates:
- results/all_5_models_master_comparison.csv
- figures/fig14_all_5_models_comparison.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PRIMARY_COLOR = "#2b5c8f"
ACCENT_COLOR = "#d95f02"

def build_master_comparison(
    results_dir: str = "results",
    figures_dir: str = "figures"
):
    trad_df = pd.read_csv(os.path.join(results_dir, "traditional_models_metrics.csv"))
    trad_ver_df = pd.read_csv(os.path.join(results_dir, "traditional_models_verified_test_metrics.csv"))
    bilstm_df = pd.read_csv(os.path.join(results_dir, "bilstm_metrics.csv"))
    trans_df = pd.read_csv(os.path.join(results_dir, "transformer_models_metrics.csv"))

    # Standardize model names
    trans_df["Model"] = trans_df["Model"].replace({
        "Distilbert": "DistilBERT",
        "Roberta": "RoBERTa"
    })

    master_df = pd.concat([trad_df, trad_ver_df, bilstm_df, trans_df], ignore_index=True)
    master_csv = os.path.join(results_dir, "all_5_models_master_comparison.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"[MasterComparison] Master comparison saved to {master_csv}")

    # Plot Fig 14: Master Comparison Bar Chart on Human-Verified Test Benchmark
    ver_df = master_df[master_df["Split"] == "Human-Verified Test"].copy()

    # Reorder models from baselines to modern transformers
    model_order = ["Linear SVM", "Logistic Regression", "BiLSTM", "DistilBERT", "RoBERTa"]
    ver_df["Model"] = pd.Categorical(ver_df["Model"], categories=model_order, ordered=True)
    ver_df = ver_df.sort_values("Model")

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    x = np.arange(len(ver_df))
    width = 0.26

    rects1 = ax.bar(x - width, ver_df["Accuracy"], width, label="Accuracy", color=PRIMARY_COLOR, alpha=0.9, edgecolor="black")
    rects2 = ax.bar(x, ver_df["Macro_F1"], width, label="Macro-F1", color=ACCENT_COLOR, alpha=0.9, edgecolor="black")
    rects3 = ax.bar(x + width, ver_df["Action_Item_F1"], width, label="Action-Item F1", color="#2ca02c", alpha=0.9, edgecolor="black")

    ax.set_ylabel("Score (0.0 - 1.0)", weight="semibold", fontsize=11)
    ax.set_title("Figure 14: Comprehensive Benchmark across All 5 Models on Human-Verified Test Data", pad=15, weight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(ver_df["Model"], weight="semibold", fontsize=11)
    ax.set_ylim(0.5, 1.02)
    ax.legend(frameon=True, loc="lower right", fontsize=10.5)

    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            val = bar.get_height()
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8.5, rotation=35, weight="bold"
            )

    plt.tight_layout()
    fig14_path = os.path.join(figures_dir, "fig14_all_5_models_comparison.png")
    plt.savefig(fig14_path, dpi=300)
    plt.close()
    print(f"[MasterComparison] Saved {fig14_path}")

    print("\n" + "=" * 75)
    print("ALL 5 MODELS MASTER EVALUATION: HUMAN-VERIFIED BENCHMARK:")
    print(ver_df[["Model", "Accuracy", "Precision", "Recall", "Action_Item_F1", "Macro_F1", "ROC_AUC"]].to_string(index=False))
    print("=" * 75)

    return master_df

if __name__ == "__main__":
    build_master_comparison()
