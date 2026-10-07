"""
Automatic Speech Recognition (ASR) Pipeline using OpenAI Whisper.
Features:
- Whisper base.en transcription with standalone soundfile audio loader
- Benchmarks Word Error Rate (WER) and Character Error Rate (CER) via jiwer
- Evaluates on the benchmark audio clips in data/audio_samples/
- Exports results to results/asr_evaluation.csv
"""

import os
import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal
from typing import Dict, List, Tuple, Optional
import whisper
import whisper.audio


def load_audio_direct(file: str, sr: int = 16000) -> np.ndarray:
    """
    Direct PCM audio loader using soundfile and scipy.
    Bypasses external ffmpeg binaries on Windows.
    """
    audio, sample_rate = sf.read(file)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != sr:
        num_samples = int(len(audio) * float(sr) / sample_rate)
        audio = scipy.signal.resample(audio, num_samples)
    return audio.astype(np.float32)


# Patch whisper's internal audio loader
whisper.audio.load_audio = load_audio_direct


class WhisperASRPipeline:
    """
    ASR Pipeline with WER and CER evaluation.
    """

    def __init__(self, model_name: str = "base.en", device: str = None):
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"[ASR] Loading OpenAI Whisper '{model_name}' on {self.device}...")
        self.model = whisper.load_model(model_name, device=self.device)

    def transcribe(self, audio_filepath: str) -> Dict:
        """Transcribes an audio file into text with timing metadata."""
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

        result = self.model.transcribe(audio_filepath, language="en", fp16=(self.device == "cuda"))
        return {
            "text": result["text"].strip(),
            "segments": result.get("segments", [])
        }

    def evaluate_benchmark_clips(
        self,
        ref_csv_path: str = "data/audio_samples/ground_truth_references.csv",
        results_dir: str = "results"
    ) -> pd.DataFrame:
        """
        Evaluates Whisper on benchmark speech clips and computes WER and CER.
        """
        os.makedirs(results_dir, exist_ok=True)
        if not os.path.exists(ref_csv_path):
            raise FileNotFoundError(f"Reference CSV not found: {ref_csv_path}")

        df_refs = pd.read_csv(ref_csv_path)
        records = []

        print(f"\n[ASR] Benchmarking Whisper on {len(df_refs)} audio clips...")
        for _, row in df_refs.iterrows():
            clip_id = row["clip_id"]
            wav_path = row["filepath"]
            ref_text = row["reference_text"]
            speaker = row["speaker"]

            res = self.transcribe(wav_path)
            hyp_text = res["text"]

            # Compute error rates
            wer_score = jiwer.wer(ref_text, hyp_text)
            cer_score = jiwer.cer(ref_text, hyp_text)

            records.append({
                "clip_id": clip_id,
                "filename": os.path.basename(wav_path),
                "speaker": speaker,
                "ground_truth": ref_text,
                "whisper_transcript": hyp_text,
                "WER": round(wer_score, 4),
                "CER": round(cer_score, 4)
            })

            print(f"[{clip_id}] WER: {wer_score:.2%} | Ref: '{ref_text}'")
            print(f"       Hyp: '{hyp_text}'")

        df_results = pd.DataFrame(records)
        out_csv = os.path.join(results_dir, "asr_evaluation.csv")
        df_results.to_csv(out_csv, index=False)

        avg_wer = df_results["WER"].mean()
        avg_cer = df_results["CER"].mean()
        print("\n" + "=" * 60)
        print(f"[ASR Evaluation Complete]")
        print(f"Average Word Error Rate (WER):      {avg_wer:.2%}")
        print(f"Average Character Error Rate (CER): {avg_cer:.2%}")
        print(f"Results saved to {out_csv}")
        print("=" * 60)

        return df_results


if __name__ == "__main__":
    asr = WhisperASRPipeline()
    asr.evaluate_benchmark_clips()
