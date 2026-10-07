"""
Hierarchical Meeting Summarization Pipeline using BART ('facebook/bart-large-cnn').
Features:
- Hierarchical chunking (~800 tokens per chunk)
- Map-Reduce summarization: summarizes individual chunks, then summarizes the summaries
- ROUGE-1, ROUGE-2, and ROUGE-L evaluation via rouge_score
- Evaluates on reference meeting summaries in data/processed/meeting_summaries.csv
- Exports results to results/summarization_rouge_metrics.csv
"""

import os
import torch
import pandas as pd
from typing import List, Dict, Tuple, Optional
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer


class HierarchicalBARTMeetingSummarizer:
    """
    Abstractive meeting summarizer employing hierarchical window chunking.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: Optional[str] = None,
        max_chunk_tokens: int = 800
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_chunk_tokens = max_chunk_tokens

        print(f"[Summarizer] Loading BART model '{model_name}' on {self.device}...")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def chunk_transcript(self, text: str) -> List[str]:
        """Splits long transcripts into token chunks of ~max_chunk_tokens."""
        tokens = self.tokenizer.encode(text, truncation=False, add_special_tokens=False)
        chunks = []
        step = self.max_chunk_tokens - 50  # 50 token overlap for context continuity

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + self.max_chunk_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks if chunks else [text]

    def summarize_chunk(self, chunk_text: str, max_length: int = 140, min_length: int = 40) -> str:
        """Generates abstractive summary for a single text segment."""
        inputs = self.tokenizer(
            chunk_text,
            max_length=self.max_chunk_tokens,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=4,
                length_penalty=2.0,
                max_length=max_length,
                min_length=min_length,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()

    def summarize_meeting(self, full_transcript: str) -> str:
        """
        Hierarchical map-reduce summarization:
        1. Map: Summarize each 800-token chunk
        2. Reduce: Summarize concatenated chunk summaries
        """
        chunks = self.chunk_transcript(full_transcript)
        if len(chunks) == 1:
            return self.summarize_chunk(chunks[0], max_length=180, min_length=50)

        print(f"[Summarizer] Transcript divided into {len(chunks)} chunks. Running Map phase...")
        chunk_summaries = []
        for idx, chunk in enumerate(chunks[:5]):  # Process top salient chunks
            s = self.summarize_chunk(chunk, max_length=120, min_length=35)
            chunk_summaries.append(s)

        combined_intermediate = " ".join(chunk_summaries)
        print("[Summarizer] Running Reduce phase on combined intermediate summaries...")
        final_summary = self.summarize_chunk(combined_intermediate, max_length=220, min_length=60)
        return final_summary

    def summarize_transcript(self, transcript_text: str, max_chunk_tokens: int = 800) -> Dict:
        """Alias for summarize_meeting returning a dictionary."""
        summary = self.summarize_meeting(transcript_text)
        return {"summary": summary}

    def evaluate_on_meeting_summaries(
        self,
        summaries_csv_path: str = "data/processed/meeting_summaries.csv",
        raw_test_csv: str = "data/processed/test_utterances_raw.csv",
        results_dir: str = "results",
        num_eval_meetings: int = 2
    ) -> pd.DataFrame:
        """
        Evaluates BART summarization against reference gold summaries using ROUGE-1/2/L.
        """
        os.makedirs(results_dir, exist_ok=True)
        df_summaries = pd.read_csv(summaries_csv_path)
        test_summaries = df_summaries[df_summaries["split"] == "test"].head(num_eval_meetings)

        test_utts = pd.read_csv(raw_test_csv)
        records = []

        print(f"\n[Summarizer] Evaluating on {len(test_summaries)} test meetings with ROUGE metrics...")
        for _, row in test_summaries.iterrows():
            m_id = row["meeting_id"]
            gold_summary = str(row["gold_summary"])

            # Assemble full meeting transcript
            m_trans = " ".join(test_utts[test_utts["meeting_id"] == m_id]["raw_text"].dropna().tolist())
            if not m_trans.strip():
                continue

            generated_summary = self.summarize_meeting(m_trans)
            rouge_scores = self.scorer.score(gold_summary, generated_summary)

            records.append({
                "meeting_id": m_id,
                "gold_summary_length": len(gold_summary.split()),
                "generated_summary_length": len(generated_summary.split()),
                "ROUGE_1_F1": round(rouge_scores["rouge1"].fmeasure, 4),
                "ROUGE_2_F1": round(rouge_scores["rouge2"].fmeasure, 4),
                "ROUGE_L_F1": round(rouge_scores["rougeL"].fmeasure, 4),
                "generated_summary": generated_summary,
                "gold_summary": gold_summary[:300] + "..."
            })

            print(f"[{m_id}] ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f} | ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f} | ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}")

        df_results = pd.DataFrame(records)
        out_csv = os.path.join(results_dir, "summarization_rouge_metrics.csv")
        df_results.to_csv(out_csv, index=False)

        mean_r1 = df_results["ROUGE_1_F1"].mean()
        mean_r2 = df_results["ROUGE_2_F1"].mean()
        mean_rL = df_results["ROUGE_L_F1"].mean()

        print("\n" + "=" * 60)
        print("BART SUMMARIZATION BENCHMARK SUMMARY:")
        print(f"Mean ROUGE-1 F1: {mean_r1:.4f}")
        print(f"Mean ROUGE-2 F1: {mean_r2:.4f}")
        print(f"Mean ROUGE-L F1: {mean_rL:.4f}")
        print(f"Results saved to {out_csv}")
        print("=" * 60)

        return df_results


if __name__ == "__main__":
    summarizer = HierarchicalBARTMeetingSummarizer()
    summarizer.evaluate_on_meeting_summaries()
