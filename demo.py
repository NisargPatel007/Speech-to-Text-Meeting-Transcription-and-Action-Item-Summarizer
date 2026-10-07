"""
End-to-End Demonstrator Script.
Pipeline:
  Audio Input (.wav) -> Whisper (Speech-to-Text) -> Cleaning/Normalization
  -> RoBERTa Action Item Classifier -> spaCy NER & Dependency Parsing (Owner, Deadline, Task)
  -> Hierarchical BART Executive Summary.

Usage:
  python demo.py --audio data/audio_samples/clip_01.wav
  python demo.py --text "I will prepare the design review slides by Friday afternoon. Councilwoman Gilmore, can you submit the revised ordinance?"
"""

import os
import sys
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.asr_pipeline import WhisperASRPipeline
from src.preprocessor import TextPreprocessor
from src.ner_extractor import ActionItemEntityExtractor
from src.summarizer import HierarchicalBARTMeetingSummarizer


def run_pipeline(
    audio_path: str = None,
    raw_text: str = None,
    roberta_model_path: str = "models/roberta_action_item",
    bart_model_name: str = "facebook/bart-large-cnn"
):
    print("=" * 78)
    print(" SPEECH-TO-TEXT MEETING TRANSCRIPTION & ACTION ITEM SUMMARIZER")
    print(" Complete NLP University Pipeline (Whisper -> RoBERTa -> spaCy -> BART)")
    print("=" * 78)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: Speech-to-Text via Whisper (if audio provided)
    if audio_path:
        print(f"\n[Stage 1/5] ASR: Transcribing Audio Input ('{audio_path}')...")
        asr = WhisperASRPipeline(model_name="base.en", device=device)
        asr_res = asr.transcribe(audio_path)
        transcript_text = asr_res["text"]
        print(f"  -> Audio Transcription Output:\n     \"{transcript_text}\"")
    elif raw_text:
        print("\n[Stage 1/5] Direct Transcript Input provided...")
        transcript_text = raw_text.strip()
    else:
        # Default meeting conversation
        transcript_text = (
            "Good morning everyone. First, we need to review the quarterly municipal budget by Friday afternoon. "
            "Councilwoman Gilmore, will you please put Council Bill 161 on the floor for consideration? "
            "I will deploy the new authentication service to the staging server tomorrow morning. "
            "Also, can you verify the industrial design specifications before next Tuesday? "
            "I'll draft the press announcement and share it with the team by five o'clock. "
            "Thank you everyone for attending, that concludes our agenda."
        )
        print("\n[Stage 1/5] Using Standard Multidisciplinary Meeting Benchmark Dialogue...")

    # Step 2: Preprocessing & Filler Removal
    print("\n[Stage 2/5] Text Preprocessing & Conversational Disfluency Cleaning...")
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.clean_text(transcript_text)
    print(f"  -> Cleaned Text: \"{cleaned[:160]}...\"")

    # Segment into sentences / utterances
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(transcript_text)
    print(f"  -> Segmented into {len(sentences)} conversational utterances.")

    # Step 3: Utterance-Level Action Item Classification via Fine-Tuned RoBERTa
    print(f"\n[Stage 3/5] Action-Item Classification via Fine-Tuned RoBERTa ({roberta_model_path})...")
    tokenizer = AutoTokenizer.from_pretrained(roberta_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(roberta_model_path).to(device)
    model.eval()

    classified_actions = []
    for idx, sent in enumerate(sentences):
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_class = int(torch.argmax(logits, dim=-1).cpu().item())
            conf = float(probs[1])

        if pred_class == 1 and conf >= 0.50:
            classified_actions.append({
                "utterance_id": f"UTT-{idx+1:03d}",
                "raw_text": sent,
                "confidence": conf
            })

    print(f"  -> Identified {len(classified_actions)} Action-Item commitments ({len(classified_actions)/max(len(sentences), 1)*100:.1f}%).")

    # Step 4: Information Extraction (Task, Owner, Deadline) via spaCy NER & Dependency Parsing
    print("\n[Stage 4/5] Information Extraction via spaCy NER & Dependency Parsing...")
    extractor = ActionItemEntityExtractor()
    extracted_items = []

    for item in classified_actions:
        parsed = extractor.parse_utterance(
            text=item["raw_text"],
            speaker="Lead Speaker",
            confidence=item["confidence"]
        )
        extracted_items.append(parsed)

    # Step 5: Abstractive Meeting Summarization via Hierarchical BART
    print("\n[Stage 5/5] Abstractive Meeting Summarization via facebook/bart-large-cnn...")
    summarizer = HierarchicalBARTMeetingSummarizer(model_name=bart_model_name, device=device)
    summary_text = summarizer.summarize_meeting(transcript_text)

    # Render Terminal Presentation UI
    print("\n" + "=" * 78)
    print("                      END-TO-END DEMONSTRATION OUTPUT")
    print("=" * 78)

    print("\n--- 1. SPEECH-TO-TEXT TRANSCRIPT ---")
    print(transcript_text.strip())

    print("\n--- 2. EXECUTIVE MEETING SUMMARY ---")
    print(summary_text.strip())

    print("\n--- 3. STRUCTURED ACTION ITEMS TABLE ---")
    if extracted_items:
        table_records = []
        for i, item in enumerate(extracted_items):
            table_records.append({
                "Task ID": f"TASK-{i+1:03d}",
                "Task Description": item["task_description"],
                "Assigned Owner": item["owner"],
                "Deadline": item["deadline"],
                "Confidence": f"{item['confidence']:.2%}"
            })
        df_table = pd.DataFrame(table_records)
        print(df_table.to_markdown(index=False))
    else:
        print("No explicit action items identified in this session.")

    print("\n" + "=" * 78)
    print(" Pipeline execution finished successfully.")
    print("=" * 78)

    # Export demo log to results/demo_sample_output.txt
    os.makedirs("results", exist_ok=True)
    with open("results/demo_sample_output.txt", "w", encoding="utf-8") as f:
        f.write("=== END-TO-END DEMO OUTPUT ===\n\n")
        f.write(f"TRANSCRIPT:\n{transcript_text}\n\n")
        f.write(f"SUMMARY:\n{summary_text}\n\n")
        f.write("ACTION ITEMS:\n")
        if extracted_items:
            f.write(pd.DataFrame(extracted_items).to_string(index=False))
        else:
            f.write("None\n")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Speech-to-Text & Action Item Summarizer Demo")
    parser.add_argument("--audio", type=str, default=None, help="Path to input audio file (.wav/.mp3)")
    parser.add_argument("--text", type=str, default=None, help="Direct transcript text")
    args = parser.parse_args()

    run_pipeline(audio_path=args.audio, raw_text=args.text)


if __name__ == "__main__":
    main()
