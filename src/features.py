"""
Feature Engineering Module for Action Item Classification.
Extracts:
1. Bag-of-Words & TF-IDF (unigrams + bigrams)
2. Chi-Square (chi2) feature selection: extracts top discriminative features per class
3. Custom linguistic features:
   - Modal verb presence and count
   - DATE/TIME entity flag
   - PERSON/LEADERSHIP entity flag
   - Token & character lengths
4. Feature combination via scipy.sparse.hstack
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from typing import Dict, List, Tuple, Optional


MODAL_VERBS = {
    "will", "shall", "should", "must", "need", "needs", "needed",
    "can", "could", "would", "may", "might", "ought", "have to"
}

TEMPORAL_PATTERNS = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|tonight|next week|noon|morning|afternoon|evening|january|february|march|april|may|june|july|august|september|october|november|december|deadline|due|by \w+)\b",
    re.IGNORECASE
)

PERSON_LEADERSHIP_PATTERNS = re.compile(
    r"\b(councilwoman|councilman|president|mayor|clerk|chair|director|chief|mr|ms|mrs|dr|staff|manager|sponsor)\b",
    re.IGNORECASE
)


class FeaturePipeline:
    """
    Constructs multimodal text representations combining statistical n-grams
    with domain-informed linguistic features.
    """

    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=3,
            sublinear_tf=True,
            norm="l2"
        )
        self.count_vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=3
        )
        self.is_fitted = False

    def extract_custom_linguistic_features(self, df: pd.DataFrame, text_col: str = "cleaned_text") -> np.ndarray:
        """
        Extracts dense matrix of linguistic flags:
        1. modal_verb_count
        2. modal_verb_flag (binary)
        3. temporal_entity_flag (binary)
        4. person_leadership_flag (binary)
        5. normalized_token_count
        """
        features_list = []

        for text in df[text_col]:
            t_str = str(text).lower()
            tokens = t_str.split()

            # 1. Modal verbs
            modal_count = sum(1 for tok in tokens if tok in MODAL_VERBS)
            modal_flag = 1.0 if modal_count > 0 else 0.0

            # 2. Temporal flag
            has_temporal = 1.0 if TEMPORAL_PATTERNS.search(t_str) else 0.0

            # 3. Person / Leadership flag
            has_person = 1.0 if PERSON_LEADERSHIP_PATTERNS.search(t_str) else 0.0

            # 4. Normalized lengths
            norm_tokens = min(len(tokens) / 30.0, 1.0)
            norm_chars = min(len(t_str) / 150.0, 1.0)

            features_list.append([modal_count, modal_flag, has_temporal, has_person, norm_tokens, norm_chars])

        return np.array(features_list, dtype=np.float32)

    def fit_transform(self, df_train: pd.DataFrame, text_col: str = "processed_text") -> csr_matrix:
        """Fits TF-IDF vectorizer and stacks custom linguistic features."""
        print(f"[FeaturePipeline] Fitting TF-IDF Vectorizer (ngram_range={self.ngram_range}, max_features={self.max_features})...")
        tfidf_sparse = self.tfidf_vectorizer.fit_transform(df_train[text_col].fillna(""))
        custom_dense = self.extract_custom_linguistic_features(df_train, text_col="cleaned_text")
        custom_sparse = csr_matrix(custom_dense)

        combined_features = hstack([tfidf_sparse, custom_sparse]).tocsr()
        self.is_fitted = True
        print(f"[FeaturePipeline] Train feature matrix shape: {combined_features.shape}")
        return combined_features

    def transform(self, df: pd.DataFrame, text_col: str = "processed_text") -> csr_matrix:
        """Transforms unseen evaluation splits using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("FeaturePipeline must be fitted prior to calling transform().")

        tfidf_sparse = self.tfidf_vectorizer.transform(df[text_col].fillna(""))
        custom_dense = self.extract_custom_linguistic_features(df, text_col="cleaned_text")
        custom_sparse = csr_matrix(custom_dense)

        combined_features = hstack([tfidf_sparse, custom_sparse]).tocsr()
        return combined_features

    def compute_chi2_feature_importance(
        self,
        df_train: pd.DataFrame,
        text_col: str = "processed_text",
        label_col: str = "weak_label",
        top_k: int = 20,
        results_dir: str = "results"
    ) -> pd.DataFrame:
        """
        Calculates Chi-Square (chi2) statistics to identify the most discriminative words/n-grams.
        Saves results to results/chi2_top_features.csv.
        """
        os.makedirs(results_dir, exist_ok=True)
        print("[FeaturePipeline] Computing Chi-Square (chi2) feature scores across classes...")

        # Compute on CountVectorizer to reflect statistical independence test requirements
        X_bow = self.count_vectorizer.fit_transform(df_train[text_col].fillna(""))
        y = df_train[label_col].values
        feature_names = np.array(self.count_vectorizer.get_feature_names_out())

        chi2_scores, p_values = chi2(X_bow, y)

        df_chi2 = pd.DataFrame({
            "feature": feature_names,
            "chi2_score": chi2_scores,
            "p_value": p_values
        }).sort_values("chi2_score", ascending=False)

        top_features = df_chi2.head(top_k).copy()
        top_features["chi2_score"] = top_features["chi2_score"].round(2)
        top_features["p_value"] = top_features["p_value"].apply(lambda p: f"{p:.2e}")

        out_csv = os.path.join(results_dir, "chi2_top_features.csv")
        top_features.to_csv(out_csv, index=False)
        print(f"[FeaturePipeline] Saved top {top_k} Chi-Square features to {out_csv}")
        return top_features

    def save_pipeline(self, output_path: str = "models/tfidf_vectorizer.joblib"):
        """Saves fitted TF-IDF vectorizer and configurations."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump({
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "ngram_range": self.ngram_range,
            "max_features": self.max_features
        }, output_path)
        print(f"[FeaturePipeline] Feature pipeline saved to {output_path}")

    @classmethod
    def load_pipeline(cls, model_path: str = "models/tfidf_vectorizer.joblib"):
        """Loads serialized feature pipeline."""
        data = joblib.load(model_path)
        instance = cls(max_features=data["max_features"], ngram_range=data["ngram_range"])
        instance.tfidf_vectorizer = data["tfidf_vectorizer"]
        instance.is_fitted = True
        return instance


if __name__ == "__main__":
    df_train = pd.read_csv("data/processed/train_preprocessed.csv")
    pipeline = FeaturePipeline(max_features=3000)
    X_train = pipeline.fit_transform(df_train)
    top_chi2 = pipeline.compute_chi2_feature_importance(df_train, top_k=15)
    print("\nTop Chi2 Features:\n", top_chi2.to_string(index=False))
    pipeline.save_pipeline()
