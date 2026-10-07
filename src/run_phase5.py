"""
Phase 5 Execution Orchestrator.
Fine-tunes DistilBERT and RoBERTa, evaluates both across all splits,
and generates the Master 5-Model Comparison Table.
"""

from src.train_transformers import run_transformers_pipeline

def main():
    print("=" * 70)
    print("PHASE 5: TRANSFORMER MODELS FINE-TUNING (DistilBERT & RoBERTa)")
    print("=" * 70)
    run_transformers_pipeline()
    print("\n[Phase 5] Execution completed successfully.")

if __name__ == "__main__":
    main()
