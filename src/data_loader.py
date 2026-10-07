"""
Data Loader Module for MeetingBank Dataset (Hugging Face: huuuyeah/meetingbank).
Enforces strict MEETING-LEVEL SPLITS (Zero dialogue/speaker leakage across train, val, and test).
Extracts utterance-level turns, speakers, and gold standard meeting summaries.
"""

import os
import re
import json
import pandas as pd
import numpy as np
import nltk
from typing import Dict, List, Tuple, Optional

# Ensure punkt is available
try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize


def extract_meeting_id(uid: str) -> str:
    """Extracts canonical meeting ID from segment UID."""
    parts = uid.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return uid


def parse_transcript_to_utterances(transcript_text: str, meeting_id: str) -> List[Dict]:
    """
    Splits meeting transcript into utterance-level units.
    Identifies speaker names where formatted (e.g. 'President Brooks: ...', 'Councilwoman Gilmore: ...').
    """
    sentences = sent_tokenize(transcript_text.strip())
    utterances = []
    current_speaker = "Council Member"
    turn_idx = 0

    # Common speaker role prefixes in council transcripts
    speaker_pattern = re.compile(
        r"^(President\s+[A-Za-z]+|Councilwoman\s+[A-Za-z]+|Councilman\s+[A-Za-z]+|Council\s+Member\s+[A-Za-z]+|Mayor\s+[A-Za-z]+|Clerk|The\s+Chair|Chair\s+[A-Za-z]+|Mr\.\s+[A-Za-z]+|Ms\.\s+[A-Za-z]+)\s*[:,]\s*(.*)$",
        re.IGNORECASE
    )

    for sent in sentences:
        sent_clean = sent.strip()
        if not sent_clean or len(sent_clean.split()) < 3:
            continue

        match = speaker_pattern.match(sent_clean)
        if match:
            current_speaker = match.group(1).title()
            utt_text = match.group(2).strip()
            if not utt_text:
                continue
        else:
            utt_text = sent_clean

        utterances.append({
            "utterance_id": f"{meeting_id}_turn_{turn_idx:05d}",
            "meeting_id": meeting_id,
            "turn_index": turn_idx,
            "speaker": current_speaker,
            "raw_text": utt_text
        })
        turn_idx += 1

    return utterances


def load_meetingbank_corpus(
    cache_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    num_train_meetings: int = 8,
    num_val_meetings: int = 2,
    num_test_meetings: int = 2,
    random_seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Loads MeetingBank from Hugging Face, enforces zero-leakage meeting-level partition,
    segments into utterances, and caches processed splits.
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    train_path = os.path.join(processed_dir, "train_utterances_raw.csv")
    val_path = os.path.join(processed_dir, "val_utterances_raw.csv")
    test_path = os.path.join(processed_dir, "test_utterances_raw.csv")
    summary_path = os.path.join(processed_dir, "meeting_summaries.csv")

    if all(os.path.exists(p) for p in [train_path, val_path, test_path, summary_path]):
        print("[DataLoader] Existing pre-split meeting datasets found on disk. Loading...")
        return {
            "train": pd.read_csv(train_path),
            "val": pd.read_csv(val_path),
            "test": pd.read_csv(test_path),
            "summaries": pd.read_csv(summary_path)
        }

    print("[DataLoader] Loading 'huuuyeah/meetingbank' dataset from Hugging Face...")
    from datasets import load_dataset
    hf_ds = load_dataset("huuuyeah/meetingbank")

    # Group all instances by canonical meeting ID
    meetings_data = {}
    for split in ["train", "validation", "test"]:
        for item in hf_ds[split]:
            m_id = extract_meeting_id(item["uid"])
            if m_id not in meetings_data:
                meetings_data[m_id] = {
                    "meeting_id": m_id,
                    "transcripts": [],
                    "summaries": []
                }
            meetings_data[m_id]["transcripts"].append(item["transcript"])
            meetings_data[m_id]["summaries"].append(item["summary"])

    all_meeting_ids = sorted(list(meetings_data.keys()))
    print(f"[DataLoader] Total unique meetings available in MeetingBank: {len(all_meeting_ids)}")

    # Enforce deterministic, leak-free partition by meeting ID
    rng = np.random.RandomState(random_seed)
    shuffled_meetings = list(all_meeting_ids)
    rng.shuffle(shuffled_meetings)

    total_needed = num_train_meetings + num_val_meetings + num_test_meetings
    selected_meetings = shuffled_meetings[:total_needed]

    train_m_ids = selected_meetings[:num_train_meetings]
    val_m_ids = selected_meetings[num_train_meetings : num_train_meetings + num_val_meetings]
    test_m_ids = selected_meetings[num_train_meetings + num_val_meetings :]

    print(f"[DataLoader] Meeting Splits (STRICT 0% LEAKAGE):")
    print(f"  - Train Meetings ({len(train_m_ids)}): {train_m_ids}")
    print(f"  - Val Meetings   ({len(val_m_ids)}): {val_m_ids}")
    print(f"  - Test Meetings  ({len(test_m_ids)}): {test_m_ids}")

    # Build utterance datasets and summary tables
    splits_dict = {"train": train_m_ids, "val": val_m_ids, "test": test_m_ids}
    result_dfs = {}
    summaries_records = []

    for split_name, m_ids in splits_dict.items():
        utts_list = []
        for m_id in m_ids:
            data = meetings_data[m_id]
            combined_transcript = "\n".join(data["transcripts"])
            combined_summary = " ".join(data["summaries"])

            summaries_records.append({
                "meeting_id": m_id,
                "split": split_name,
                "gold_summary": combined_summary
            })

            m_utts = parse_transcript_to_utterances(combined_transcript, m_id)
            utts_list.extend(m_utts)

        df = pd.DataFrame(utts_list)
        result_dfs[split_name] = df
        out_p = os.path.join(processed_dir, f"{split_name}_utterances_raw.csv")
        df.to_csv(out_p, index=False)
        print(f"[DataLoader] {split_name.upper()} set created: {len(df)} utterances from {len(m_ids)} meetings.")

    df_summaries = pd.DataFrame(summaries_records)
    df_summaries.to_csv(summary_path, index=False)
    result_dfs["summaries"] = df_summaries

    return result_dfs


if __name__ == "__main__":
    dfs = load_meetingbank_corpus()
    for k, v in dfs.items():
        print(f"{k}: shape {v.shape}")
