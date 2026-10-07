"""
Phase 1 Weak Labeling Execution Script.
Applies linguistic rules to Train, Val, and Test splits.
Generates:
1. data/processed/train_labeled.csv
2. data/processed/val_labeled.csv
3. data/processed/test_labeled.csv
4. data/processed/manual_verification_test_200.csv (Human ground truth verification benchmark)
"""

import os
import pandas as pd
from src.weak_labeler import ActionItemWeakLabeler


def main():
    print("[Phase 1] Initializing ActionItemWeakLabeler with spaCy 'en_core_web_sm'...")
    labeler = ActionItemWeakLabeler(spacy_model="en_core_web_sm")

    processed_dir = "data/processed"
    train_path = os.path.join(processed_dir, "train_utterances_raw.csv")
    val_path = os.path.join(processed_dir, "val_utterances_raw.csv")
    test_path = os.path.join(processed_dir, "test_utterances_raw.csv")

    print("[Phase 1] Loading raw splits...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"[Phase 1] Labeling Train Set ({len(train_df)} utterances)...")
    train_labeled = labeler.label_dataframe(train_df)
    train_labeled.to_csv(os.path.join(processed_dir, "train_labeled.csv"), index=False)
    n_pos_train = (train_labeled["weak_label"] == 1).sum()
    print(f"  -> Train positive action items: {n_pos_train}/{len(train_labeled)} ({n_pos_train/len(train_labeled)*100:.2f}%)")

    print(f"[Phase 1] Labeling Val Set ({len(val_df)} utterances)...")
    val_labeled = labeler.label_dataframe(val_df)
    val_labeled.to_csv(os.path.join(processed_dir, "val_labeled.csv"), index=False)
    n_pos_val = (val_labeled["weak_label"] == 1).sum()
    print(f"  -> Val positive action items: {n_pos_val}/{len(val_labeled)} ({n_pos_val/len(val_labeled)*100:.2f}%)")

    print(f"[Phase 1] Labeling Test Set ({len(test_df)} utterances)...")
    test_labeled = labeler.label_dataframe(test_df)
    test_labeled.to_csv(os.path.join(processed_dir, "test_labeled.csv"), index=False)
    n_pos_test = (test_labeled["weak_label"] == 1).sum()
    print(f"  -> Test positive action items: {n_pos_test}/{len(test_labeled)} ({n_pos_test/len(test_labeled)*100:.2f}%)")

    print("[Phase 1] Constructing 200-utterance human-verified evaluation test benchmark...")
    verified_csv = os.path.join(processed_dir, "manual_verification_test_200.csv")
    verified_df = labeler.create_verified_test_subset(
        test_df=test_labeled,
        output_csv_path=verified_csv,
        n_samples=200,
        random_seed=42
    )

    print("[Phase 1] Complete! Summary of generated datasets:")
    print(f"  - train_labeled.csv: {train_labeled.shape}")
    print(f"  - val_labeled.csv: {val_labeled.shape}")
    print(f"  - test_labeled.csv: {test_labeled.shape}")
    print(f"  - manual_verification_test_200.csv: {verified_df.shape}")


if __name__ == "__main__":
    main()
