"""
Phase 2 Execution Orchestrator.
1. Benchmarks Stemming vs Lemmatization and logs results/preprocessing_stem_vs_lemma.csv
2. Applies comprehensive preprocessing to train, val, test, and human-verified sets
3. Executes complete EDA generating Fig 1 through Fig 6 in figures/
4. Exports structured observations to results/eda_observations.json
"""

import os
import sys
import pandas as pd
from src.preprocessor import TextPreprocessor
from src.eda import MeetingEDA


def main():
    print("=" * 70)
    print("PHASE 2: TEXT PREPROCESSING & EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    processed_dir = "data/processed"
    results_dir = "results"
    figures_dir = "figures"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Initialize Preprocessor
    print("\n[Step 2.1] Initializing Preprocessor...")
    prep = TextPreprocessor(spacy_model="en_core_web_sm")

    # Load labeled datasets
    train_labeled = pd.read_csv(os.path.join(processed_dir, "train_labeled.csv"))
    val_labeled = pd.read_csv(os.path.join(processed_dir, "val_labeled.csv"))
    test_labeled = pd.read_csv(os.path.join(processed_dir, "test_labeled.csv"))
    verified_test = pd.read_csv(os.path.join(processed_dir, "manual_verification_test_200.csv"))

    # 2. Benchmarking: Stemming vs Lemmatization
    print("\n[Step 2.2] Conducting Comparative Study: NLTK Stemming vs. spaCy Lemmatization...")
    benchmark_samples = train_labeled["raw_text"].dropna().sample(n=min(1000, len(train_labeled)), random_state=42).tolist()
    df_comparison = prep.compare_stemming_vs_lemmatization(benchmark_samples)

    comparison_csv = os.path.join(results_dir, "preprocessing_stem_vs_lemma.csv")
    df_comparison.to_csv(comparison_csv, index=False)
    print(f"Comparison saved to {comparison_csv}:")
    print(df_comparison.to_string(index=False))

    # 3. Applying Preprocessing Pipeline across all splits
    print("\n[Step 2.3] Preprocessing all splits (filler removal, lowercasing, modal retention)...")
    train_prep = prep.process_dataframe(train_labeled)
    val_prep = prep.process_dataframe(val_labeled)
    test_prep = prep.process_dataframe(test_labeled)
    verified_prep = prep.process_dataframe(verified_test)

    train_prep.to_csv(os.path.join(processed_dir, "train_preprocessed.csv"), index=False)
    val_prep.to_csv(os.path.join(processed_dir, "val_preprocessed.csv"), index=False)
    test_prep.to_csv(os.path.join(processed_dir, "test_preprocessed.csv"), index=False)
    verified_prep.to_csv(os.path.join(processed_dir, "manual_verification_preprocessed.csv"), index=False)

    print(f"  -> Train preprocessed: {train_prep.shape} saved.")
    print(f"  -> Val preprocessed:   {val_prep.shape} saved.")
    print(f"  -> Test preprocessed:  {test_prep.shape} saved.")
    print(f"  -> Verified test prep: {verified_prep.shape} saved.")

    # 4. Exploratory Data Analysis (EDA)
    print("\n[Step 2.4] Executing Exploratory Data Analysis and generating figures...")
    eda = MeetingEDA(figures_dir=figures_dir, results_dir=results_dir)

    print("  -> Generating Fig 1: Class Balance...")
    obs1 = eda.plot_class_balance(train_prep)

    print("  -> Generating Fig 2: Utterance Length Distribution...")
    obs2 = eda.plot_utterance_length_distribution(train_prep)

    print("  -> Generating Fig 3: Word Clouds per Class...")
    obs3 = eda.plot_wordclouds(train_prep)

    print("  -> Generating Fig 4: Top n-grams...")
    obs4 = eda.plot_top_ngrams(train_prep)

    print("  -> Generating Fig 5: Speaker Contribution...")
    obs5 = eda.plot_speaker_contribution(train_prep)

    print("  -> Generating Fig 6: Action Item Temporal Density...")
    obs6 = eda.plot_temporal_density(train_prep)

    obs_file = eda.save_all_observations()

    print("\n" + "=" * 70)
    print(f"PHASE 2 EXECUTION COMPLETE!")
    print(f"All 6 figures saved in '{figures_dir}/'.")
    print(f"Metrics & observations recorded in '{results_dir}/'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
