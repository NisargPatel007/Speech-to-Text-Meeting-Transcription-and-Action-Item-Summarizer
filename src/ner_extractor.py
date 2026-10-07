"""
Information Extraction Module for Action Items.
Uses spaCy Named Entity Recognition (NER) & Dependency Parsing to extract:
1. Task Action Phrase (syntactic verb + object dependency extraction)
2. Assigned Owner (PERSON / ORG entities + speaker turn resolution)
3. Deadline / Milestone (DATE / TIME entities + prepositional temporal parsing)
"""

import os
import re
import spacy
import pandas as pd
from typing import Dict, List, Tuple, Optional


class ActionItemEntityExtractor:
    """
    Extracts structured action item metadata from classified utterances.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        print(f"[NER Extractor] Loading spaCy pipeline ('{spacy_model}')...")
        self.nlp = spacy.load(spacy_model)

        self.temporal_preps = re.compile(
            r"\b(by|before|until|due|due on|prior to)\s+([a-zA-Z0-9_\s]+?)(?:[\.\,\;\!]|$)",
            re.IGNORECASE
        )

        self.first_person_cues = re.compile(r"\b(i\s+will|i'll|i\s+am\s+going\s+to|i'm\s+going\s+to|i\s+can|i\s+must|i\s+need\s+to|i'll\s+take\s+care)\b", re.IGNORECASE)
        self.collective_cues = re.compile(r"\b(we\s+will|we'll|we\s+need\s+to|we\s+must|let's|let\s+us)\b", re.IGNORECASE)
        self.second_person_cues = re.compile(r"\b(can\s+you|could\s+you|would\s+you|you\s+should|you\s+need\s+to|please)\b", re.IGNORECASE)

    def extract_deadline(self, doc: spacy.tokens.Doc, text: str) -> str:
        """Extracts date/time milestones using NER and prepositional patterns."""
        # 1. Search for DATE or TIME entities
        date_ents = [ent.text.strip() for ent in doc.ents if ent.label_ in ("DATE", "TIME")]
        if date_ents:
            return ", ".join(date_ents)

        # 2. Check prepositional time phrases (e.g. 'by Friday afternoon', 'before next week')
        match = self.temporal_preps.search(text)
        if match:
            candidate = match.group(0).strip(" .,;!")
            if len(candidate.split()) <= 5:
                return candidate

        return "Not Specified (Ongoing)"

    def extract_owner(self, doc: spacy.tokens.Doc, speaker: str, text: str) -> str:
        """Resolves task owner using NER, speaker turns, and speech act pronouns."""
        # 1. Check for explicit PERSON entities in the utterance
        person_ents = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]

        # If addressed directive with named person (e.g., 'Councilwoman Gilmore, will you put bill...')
        if person_ents and self.second_person_cues.search(text):
            return person_ents[0]

        # 2. First-person commitment -> Speaker is the owner
        if self.first_person_cues.search(text):
            return speaker if speaker and speaker != "Unknown" else "Speaker (First Person)"

        # 3. Collective / joint commitment -> Joint ownership
        if self.collective_cues.search(text):
            return "Joint Team / Working Group"

        # 4. Directive request with person entity
        if person_ents:
            return person_ents[0]

        # 5. Default delegation
        if self.second_person_cues.search(text):
            return "Designated Assignee"

        return speaker if speaker and speaker != "Unknown" else "Designated Assignee"

    def extract_task_action(self, doc: spacy.tokens.Doc, text: str) -> str:
        """
        Isolates the core action verb and object phrase using dependency parsing.
        """
        # Find root verb or modal auxiliary complement
        root = [tok for tok in doc if tok.pos_ in ("VERB", "AUX") and tok.dep_ in ("ROOT", "xcomp", "ccomp")]
        if not root:
            # Fallback: first verb found
            root = [tok for tok in doc if tok.pos_ == "VERB"]

        if root:
            main_verb = root[0]
            # Collect the subtree starting from this verb, stripping leading conversational junk
            verb_subtree = [t.text for t in main_verb.subtree]
            task_str = " ".join(verb_subtree).strip(" .,;!")
            # Clean up leading pronouns if present
            clean_task = re.sub(r"^(i|we|you|they|he|she)\s+(will|must|need to|shall|can|could|should|would)?\s*", "", task_str, flags=re.IGNORECASE)
            if len(clean_task.split()) >= 3:
                return clean_task[:100]

        # Fallback to normalized utterance
        return text.strip(" .,;!")[:100]

    def parse_utterance(self, text: str, speaker: str = "Unknown", confidence: float = 1.0) -> Dict:
        """Parses an action-item utterance into structured fields."""
        doc = self.nlp(text)
        task = self.extract_task_action(doc, text)
        owner = self.extract_owner(doc, speaker, text)
        deadline = self.extract_deadline(doc, text)

        return {
            "task_description": task,
            "owner": owner,
            "deadline": deadline,
            "raw_text": text,
            "speaker": speaker,
            "confidence": round(float(confidence), 3)
        }

    def extract_from_utterance(self, text: str, speaker_turn: str = "Lead Speaker", confidence: float = 1.0) -> Dict:
        """Alias for parse_utterance to support alternative call signatures."""
        return self.parse_utterance(text=text, speaker=speaker_turn, confidence=confidence)

    def process_action_item_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "cleaned_text",
        speaker_col: str = "speaker",
        conf_col: str = "confidence",
        output_csv: str = "results/action_items_extracted.csv"
    ) -> pd.DataFrame:
        """Extracts structured entities from an entire dataframe of action items."""
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results = []

        for idx, row in df.iterrows():
            text = str(row.get(text_col, ""))
            speaker = str(row.get(speaker_col, "Unknown"))
            conf = float(row.get(conf_col, 1.0))

            parsed = self.parse_utterance(text=text, speaker=speaker, confidence=conf)
            parsed["task_id"] = f"TASK-{idx+1:04d}"
            results.append(parsed)

        out_df = pd.DataFrame(results)
        cols_order = ["task_id", "task_description", "owner", "deadline", "confidence", "speaker", "raw_text"]
        out_df = out_df[[c for c in cols_order if c in out_df.columns]]
        out_df.to_csv(output_csv, index=False)
        print(f"[NER Extractor] Extracted {len(out_df)} action items. Saved to {output_csv}")
        return out_df


if __name__ == "__main__":
    extractor = ActionItemEntityExtractor()
    sample_commitments = [
        ("We need to finalize the quarterly budget review by Friday afternoon.", "Project Manager"),
        ("I will deploy the new authentication service to the staging server tomorrow morning.", "Lead Developer"),
        ("Councilwoman Gilmore, will you please put Council Bill 161 on the floor before next Tuesday?", "President Brooks"),
        ("I'll draft the press announcement and share it with the team by five o'clock.", "Marketing Lead")
    ]
    for text, spk in sample_commitments:
        res = extractor.parse_utterance(text, speaker=spk, confidence=0.98)
        print(f"Task:     {res['task_description']}")
        print(f"Owner:    {res['owner']}")
        print(f"Deadline: {res['deadline']}")
        print("-" * 50)
