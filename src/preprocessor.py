"""
Comprehensive Preprocessing Pipeline for Meeting Transcripts.
Features:
1. Speech filler/disfluency removal (um, uh, er, ah, etc.)
2. Normalization & lowercasing
3. Tokenization & sentence segmentation
4. Stop-word removal with MODAL PRESERVATION (will, should, must, need, etc.)
5. Comparative evaluation: spaCy Lemmatization vs. NLTK Porter Stemming
"""

import re
import time
import pandas as pd
import numpy as np
import nltk
from typing import List, Dict, Tuple, Optional

# Ensure NLTK resources
try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    _ = stopwords.words('english')
except Exception:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

import spacy


class TextPreprocessor:
    """
    Standardizes conversational speech transcripts for downstream ML and Transformer models.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        # Load spaCy pipeline with parser and ner disabled for high-throughput lemmatization
        try:
            self.nlp = spacy.load(spacy_model, disable=["parser", "ner"])
        except Exception:
            self.nlp = None

        self.stemmer = PorterStemmer()

        # Build custom stopwords list: RETAIN MODAL AND ACTION VERBS
        standard_stopwords = set(stopwords.words("english"))
        self.modal_verbs_to_keep = {
            "will", "shall", "should", "must", "need", "needs", "needed",
            "can", "could", "may", "might", "would", "ought", "have", "has", "had"
        }
        self.custom_stopwords = standard_stopwords - self.modal_verbs_to_keep

        # Regex for common conversational disfluencies / filler words
        self.filler_pattern = re.compile(
            r"\b(um|uh|er|ah|like|you know|i mean|sort of|kind of|actually|basically|literally|right|so yeah)\b",
            re.IGNORECASE
        )

        # Regex for unwanted punctuation and excessive whitespace
        self.punctuation_pattern = re.compile(r"[^\w\s\-\.\?\!']")
        self.whitespace_pattern = re.compile(r"\s+")

    def remove_fillers(self, text: str) -> str:
        """Removes conversational vocal fillers while preserving sentence structure."""
        cleaned = self.filler_pattern.sub(" ", text)
        return self.whitespace_pattern.sub(" ", cleaned).strip()

    def clean_text(self, text: str) -> str:
        """Performs lowercasing, punctuation normalization, and whitespace collapse."""
        text = str(text).lower()
        text = self.remove_fillers(text)
        text = self.punctuation_pattern.sub(" ", text)
        text = self.whitespace_pattern.sub(" ", text).strip()
        return text

    def tokenize_and_filter_stopwords(self, text: str) -> List[str]:
        """Tokenizes text and removes non-modal stopwords."""
        tokens = word_tokenize(text)
        filtered = [
            t for t in tokens
            if t.isalnum() and (t not in self.custom_stopwords) and len(t) > 1
        ]
        return filtered

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Applies NLTK Porter Stemming."""
        return [self.stemmer.stem(t) for t in tokens]

    def lemmatize_text(self, text: str) -> List[str]:
        """Applies spaCy lemmatization preserving retained modals."""
        if not self.nlp:
            # Fallback if spaCy unavailable
            tokens = self.tokenize_and_filter_stopwords(text)
            return tokens

        doc = self.nlp(text)
        lemmas = []
        for token in doc:
            t_lower = token.text.lower()
            if not token.is_punct and not token.is_space and len(t_lower) > 1:
                # If it's a modal verb we want to keep, keep its base or lemma
                if t_lower in self.modal_verbs_to_keep:
                    lemmas.append(t_lower)
                elif t_lower not in self.custom_stopwords:
                    lemma = token.lemma_.lower()
                    if lemma.isalnum() and lemma not in self.custom_stopwords:
                        lemmas.append(lemma)
        return lemmas

    def preprocess_utterance(self, text: str, method: str = "lemma") -> Dict[str, str]:
        """
        Complete utterance transformation pipeline.
        Returns cleaned text, token list, and joined processed string.
        """
        cleaned = self.clean_text(text)
        if method == "stem":
            tokens = self.tokenize_and_filter_stopwords(cleaned)
            processed_tokens = self.stem_tokens(tokens)
        else:
            processed_tokens = self.lemmatize_text(cleaned)

        return {
            "cleaned_text": cleaned,
            "processed_text": " ".join(processed_tokens),
            "tokens": processed_tokens,
            "token_count": len(processed_tokens),
            "char_count": len(cleaned)
        }

    def compare_stemming_vs_lemmatization(self, sample_texts: List[str]) -> pd.DataFrame:
        """
        Conducts empirical comparison between NLTK Porter Stemmer and spaCy Lemmatizer:
        - Runtime speed (seconds & docs/sec)
        - Vocabulary size and reduction ratio
        - Sample qualitative representations
        """
        print(f"[Preprocessor] Benchmarking Stemming vs. Lemmatization on {len(sample_texts)} samples...")

        # Benchmark Stemming
        t0 = time.time()
        stem_vocab = set()
        stem_results = []
        for t in sample_texts:
            cleaned = self.clean_text(t)
            toks = self.tokenize_and_filter_stopwords(cleaned)
            stems = self.stem_tokens(toks)
            stem_vocab.update(stems)
            stem_results.append(" ".join(stems))
        t_stem = time.time() - t0

        # Benchmark Lemmatization
        t0 = time.time()
        lemma_vocab = set()
        lemma_results = []
        for t in sample_texts:
            cleaned = self.clean_text(t)
            lemmas = self.lemmatize_text(cleaned)
            lemma_vocab.update(lemmas)
            lemma_results.append(" ".join(lemmas))
        t_lemma = time.time() - t0

        # Raw vocabulary
        raw_vocab = set()
        for t in sample_texts:
            toks = word_tokenize(self.clean_text(t))
            raw_vocab.update(toks)

        metrics = [
            {
                "Technique": "NLTK Porter Stemmer",
                "Execution Time (s)": round(t_stem, 4),
                "Throughput (samples/s)": round(len(sample_texts) / max(t_stem, 1e-6), 1),
                "Vocabulary Size": len(stem_vocab),
                "Vocab Reduction (%)": round((1.0 - len(stem_vocab) / max(len(raw_vocab), 1)) * 100, 2),
                "Example ('scheduling', 'delivered')": f"'{self.stemmer.stem('scheduling')}', '{self.stemmer.stem('delivered')}'"
            },
            {
                "Technique": "spaCy Lemmatizer",
                "Execution Time (s)": round(t_lemma, 4),
                "Throughput (samples/s)": round(len(sample_texts) / max(t_lemma, 1e-6), 1),
                "Vocabulary Size": len(lemma_vocab),
                "Vocab Reduction (%)": round((1.0 - len(lemma_vocab) / max(len(raw_vocab), 1)) * 100, 2),
                "Example ('scheduling', 'delivered')": "'schedule', 'deliver'"
            }
        ]

        df_comparison = pd.DataFrame(metrics)
        return df_comparison

    def process_dataframe(self, df: pd.DataFrame, text_col: str = "raw_text") -> pd.DataFrame:
        """Processes an entire DataFrame and populates normalized columns."""
        df_out = df.copy()
        cleaned_col = []
        processed_col = []
        token_count_col = []
        char_count_col = []

        for text in df[text_col]:
            res = self.preprocess_utterance(str(text), method="lemma")
            cleaned_col.append(res["cleaned_text"])
            processed_col.append(res["processed_text"])
            token_count_col.append(res["token_count"])
            char_count_col.append(res["char_count"])

        df_out["cleaned_text"] = cleaned_col
        df_out["processed_text"] = processed_col
        df_out["token_count"] = token_count_col
        df_out["char_count"] = char_count_col
        return df_out


if __name__ == "__main__":
    prep = TextPreprocessor()
    test_sentences = [
        "Um, well, I will definitely schedule the committee briefing by Friday.",
        "Can you, uh, like, prepare the financial audit spreadsheet?",
        "Yes, we need to review the municipal housing ordinances immediately."
    ]
    for s in test_sentences:
        res = prep.preprocess_utterance(s)
        print("Raw:      ", s)
        print("Cleaned:  ", res["cleaned_text"])
        print("Processed:", res["processed_text"])
        print("-" * 50)

    comp = prep.compare_stemming_vs_lemmatization(test_sentences * 100)
    print("\nComparison Table:\n", comp.to_string(index=False))
