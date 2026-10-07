"""
Phase 3 Execution Orchestrator.
1. Extracts TF-IDF + linguistic features
2. Calculates Chi-Square feature ranking
3. Trains and tunes Logistic Regression and Linear SVM
4. Computes metrics on Validation, Weak Test, and Human-Verified Benchmark
5. Generates Confusion Matrix heatmaps (Fig 7 and Fig 8)
"""

from src.train_traditional import train_and_evaluate_traditional_models

def main():
    print("=" * 70)
    print("PHASE 3: FEATURE ENGINEERING & TRADITIONAL MACHINE LEARNING")
    print("=" * 70)
    train_and_evaluate_traditional_models()
    print("\n[Phase 3] Execution finished successfully.")

if __name__ == "__main__":
    main()
