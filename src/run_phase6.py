"""
Phase 6 Execution Orchestrator.
1. Whisper ASR: Evaluates on 8 benchmark speech clips and computes WER/CER
2. Information Extraction: Extracts tasks, owners, deadlines via spaCy NER & Dependency Parsing
3. Abstractive Summarization: Evaluates BART-large-CNN on meeting transcripts via ROUGE-1/2/L
4. Demonstrator: Executes end-to-end demo on audio clip and multi-turn meeting dialogue
"""

import os
import pandas as pd
from src.asr_pipeline import WhisperASRPipeline
from src.ner_extractor import ActionItemEntityExtractor
from src.summarizer import HierarchicalBARTMeetingSummarizer
from demo import run_pipeline


def main():
    print("=" * 70)
    print("PHASE 6: SPEECH-TO-TEXT (WHISPER), SUMMARIZATION (BART) & END-TO-END DEMO")
    print("=" * 70)

    # 1. Automatic Speech Recognition (Whisper WER Benchmarking)
    print("\n[Step 6.1] Executing Whisper ASR Evaluation on 8 Speech Clips...")
    asr = WhisperASRPipeline(model_name="base.en")
    asr_results = asr.evaluate_benchmark_clips(
        ref_csv_path="data/audio_samples/ground_truth_references.csv",
        results_dir="results"
    )

    # 2. Information Extraction (NER + Dependency Parsing)
    print("\n[Step 6.2] Running NER & Dependency Parsing Entity Extraction...")
    extractor = ActionItemEntityExtractor()
    test_df = pd.read_csv("data/processed/manual_verification_preprocessed.csv")
    action_items_subset = test_df[test_df["verified_label"] == 1].copy()

    extracted_df = extractor.process_action_item_dataframe(
        df=action_items_subset,
        text_col="cleaned_text",
        speaker_col="speaker",
        conf_col="confidence",
        output_csv="results/action_items_extracted.csv"
    )
    print(f"Extracted {len(extracted_df)} action items. Sample records:")
    print(extracted_df[["task_id", "task_description", "owner", "deadline"]].head(4).to_string(index=False))

    # 3. Hierarchical BART Summarization & ROUGE Evaluation
    print("\n[Step 6.3] Running BART-large-CNN Hierarchical Summarization & ROUGE Benchmarking...")
    summarizer = HierarchicalBARTMeetingSummarizer()
    rouge_df = summarizer.evaluate_on_meeting_summaries(
        summaries_csv_path="data/processed/meeting_summaries.csv",
        raw_test_csv="data/processed/test_utterances_raw.csv",
        results_dir="results",
        num_eval_meetings=2
    )

    # 4. End-to-End Pipeline Demonstration
    print("\n[Step 6.4] Executing Complete End-to-End Demonstration...")
    run_pipeline(audio_path="data/audio_samples/clip_01.wav")

    print("\n" + "=" * 70)
    print("PHASE 6 COMPLETE! All ASR, NER, BART, and Demo deliverables generated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
