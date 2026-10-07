"""
Phase 4 Execution Orchestrator.
Trains BiLSTM on PyTorch with CUDA acceleration, logs loss & accuracy curves,
and records benchmark metrics across Validation, Weak Test, and Verified Test splits.
"""

from src.train_bilstm import train_bilstm_model

def main():
    print("=" * 70)
    print("PHASE 4: DEEP LEARNING - BIDIRECTIONAL LSTM (BiLSTM)")
    print("=" * 70)
    train_bilstm_model()
    print("\n[Phase 4] Execution completed successfully.")

if __name__ == "__main__":
    main()
