# Speech-to-Text Meeting Transcription and Action Item Summarizer

[![Python 3.10](https://img.shields.io/badge/Python-3.10.9-blue.svg)](https://www.python.org/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1+cu121-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Dataset: MeetingBank](https://img.shields.io/badge/Dataset-huuuyeah%2Fmeetingbank-green.svg)](https://huggingface.co/datasets/huuuyeah/meetingbank)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, multi-stage Natural Language Processing pipeline and research benchmark that transcribes multi-speaker conversational audio, filters conversational disfluencies, classifies dialogue commitments (action items), extracts structured task triples (Task, Owner, Deadline), and generates executive abstractive meeting minutes.

Developed as a rigorous, university-level NLP Capstone Project with 100% local reproducibility on consumer GPU hardware (NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Raw Speech Audio .wav] --> B[Stage 1: ASR Engine\nOpenAI Whisper base.en]
    B --> C[Verbatim Transcript]
    C --> D[Stage 2: Text Preprocessor\nDisfluency Pruning & Modal Preservation]
    D --> E[Cleaned Dialogue Utterances]
    E --> F[Stage 3: Action-Item Classifier\nFine-Tuned RoBERTa / DistilBERT / BiLSTM / SVM / LogReg]
    F -->|Action Items| G[Stage 4: Information Extractor\nspaCy NER & Dependency Subtree Parsing]
    F -->|Full Transcript| H[Stage 5: Abstractive Summarizer\nfacebook/bart-large-cnn Hierarchical Map-Reduce]
    G --> I[Structured Action Items Table\nTask | Owner | Deadline | Confidence]
    H --> J[Executive Meeting Summary\nConcise Minutes of Meeting]
```

---

## 📊 Dataset & Experimental Rigor

We utilize the **MeetingBank** corpus ([`huuuyeah/meetingbank`](https://huggingface.co/datasets/huuuyeah/meetingbank)), containing multi-hour municipal council meetings across six major US metropolitan bodies (Seattle, Denver, Boston, Long Beach, etc.).

### Zero Data Leakage Partitioning
To prevent subtle cross-utterance data leakage (speaker habits, council-specific terminology), we enforce strict **meeting-level stratification**:
- **Training Set**: 8 complete council meetings (9,721 dialogue turns)
- **Validation Set**: 2 complete council meetings (3,341 dialogue turns)
- **Weak Test Set**: 2 complete council meetings (1,405 dialogue turns)
- **Leakage Rate**: **0.00%** (zero meeting or speaker overlap across sets)

### The 200-Utterance Human-Verified Ground-Truth Benchmark
To honestly evaluate rule memorization vs. true semantic generalization, we constructed an isolated, human-verified ground-truth test benchmark (`data/processed/manual_verification_test_200.csv`) with 98 true action items and 102 non-action dialogue turns (51% / 49% balanced).

---

## 🏆 Model Leaderboard & Experimental Results

### 1. Action-Item Classification Benchmark
Performance on the **200-Utterance Human-Verified Ground-Truth Benchmark**:

| Model Family | Model Architecture | Accuracy | Precision | Recall | Action F1 | Macro-F1 | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Traditional ML | Linear SVM | 0.7250 | 0.8413 | 0.5408 | 0.6584 | **0.7141** | 0.8447 |
| Traditional ML | Logistic Regression ($L_2$) | 0.7650 | 0.7742 | 0.7347 | 0.7539 | **0.7645** | 0.8458 |
| Deep Learning | BiLSTM (64 hidden) | 0.7700 | 0.7708 | 0.7551 | 0.7629 | **0.7698** | 0.8514 |
| Transformer | DistilBERT (`distilbert-base-uncased`) | 0.9350 | 0.9670 | 0.8980 | 0.9312 | **0.9348** | 0.9660 |
| Transformer | **RoBERTa (`roberta-base`)** 🥇 | **0.9450** | **0.9780** | **0.9082** | **0.9418** | **0.9448** | **0.9600** |

> **Key Scientific Discovery**: Linear SVM collapsed on the human-verified benchmark (Recall fell to 54.08%) because it memorized regex heuristics from weak supervision. Conversely, **RoBERTa maintained 94.48% Macro-F1**, proving that deep contextual attention captures true semantic commitment across conversational variations.

### 2. ASR Acoustic Benchmark (OpenAI Whisper `base.en`)
Evaluated across 8 multi-speaker speech audio clips:
- **Mean Word Error Rate (WER)**: **3.78%**
- **Mean Character Error Rate (CER)**: **0.96%**
- **Perfect Zero-WER Clips**: 5 out of 8 clips (62.5%)

### 3. Abstractive Summarization Benchmark (`facebook/bart-large-cnn`)
Evaluated on full municipal council transcripts using hierarchical map-reduce chunking:
- **Mean ROUGE-1 F1**: **0.1334**
- **Mean ROUGE-2 F1**: **0.0581**
- **Mean ROUGE-L F1**: **0.1072**

---

## 📁 Repository Structure

```
NLP PROJECT/
├── data/
│   ├── audio_samples/               # 8 benchmark WAV speech clips
│   ├── processed/                   # Preprocessed CSV splits & 200-utterance ground truth
│   └── raw/                         # Raw ingested meeting splits
├── figures/                         # 14 High-resolution publication figures (PNG)
│   ├── fig1_class_balance.png
│   ├── fig2_utterance_length_hist.png
│   ├── fig3_wordclouds_by_class.png
│   ├── fig4_top_ngrams.png
│   ├── fig5_speaker_contribution.png
│   ├── fig6_action_item_temporal_density.png
│   ├── fig7_confusion_matrix_logreg.png
│   ├── fig8_confusion_matrix_svm.png
│   ├── fig9_bilstm_loss_curve.png
│   ├── fig10_bilstm_f1_accuracy_curve.png
│   ├── fig11_confusion_matrix_bilstm.png
│   ├── fig12_confusion_matrix_distilbert.png
│   ├── fig13_confusion_matrix_roberta.png
│   └── fig14_all_5_models_comparison.png
├── models/                          # Serialized model weights & vocabularies
│   ├── logistic_regression.joblib
│   ├── linear_svm.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── bilstm_classifier.pt
│   ├── bilstm_vocab.json
│   ├── distilbert_action_item/      # Fine-tuned DistilBERT weights & config
│   └── roberta_action_item/         # Fine-tuned RoBERTa weights & config
├── notebooks/
│   └── NLP_Project_Full_Pipeline.ipynb  # Complete runnable Jupyter notebook
├── presentation/
│   └── NLP_Project_Presentation.pptx    # 12-slide academic deck
├── report/
│   └── NLP_Project_Report.docx          # 35-page IEEE academic capstone report
├── results/                         # 14 Empirical metric CSVs and output logs
│   ├── all_5_models_master_comparison.csv
│   ├── asr_evaluation.csv
│   ├── action_items_extracted.csv
│   ├── summarization_rouge_metrics.csv
│   ├── preprocessing_stem_vs_lemma.csv
│   └── chi2_top_features.csv
├── src/                             # Modular Python source code
│   ├── asr_pipeline.py              # Whisper ASR & WER/CER evaluator
│   ├── preprocessor.py              # Disfluency cleaner & modal preserver
│   ├── features.py                  # TF-IDF & dense linguistic extractor
│   ├── train_bilstm.py              # PyTorch BiLSTM training loop
│   ├── train_transformers.py        # Hugging Face Trainer fine-tuner
│   ├── ner_extractor.py             # spaCy NER & dependency parsing
│   ├── summarizer.py                # BART hierarchical summarizer
│   ├── generate_report.py           # Word report builder
│   ├── generate_presentation.py     # PowerPoint deck builder
│   └── generate_notebook.py         # Jupyter notebook builder
├── demo.py                          # Unified CLI end-to-end demonstration script
├── requirements.txt                 # Exact pinned dependency specifications
└── README.md                        # Master project documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/nlp-meeting-action-summarizer.git
cd nlp-meeting-action-summarizer

# Install dependencies
pip install -r requirements.txt

# Download required spaCy and NLTK language models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### 2. Run the Live End-to-End Demo
Transcribe an audio clip, detect action items, extract assignees/deadlines, and generate an executive summary:
```bash
python demo.py --audio data/audio_samples/clip_01.wav
```

### 3. Reproduce All Pipeline Phases
To reproduce the experiment from scratch:
```bash
# Phase 1: Ingest MeetingBank & create strict partitions
python -m src.run_phase1

# Phase 2: Preprocess dialogue & generate EDA figures 1-6
python -m src.run_phase2

# Phase 3: Extract TF-IDF + dense features & train LogReg and SVM
python -m src.train_traditional

# Phase 4: Train PyTorch BiLSTM with early stopping
python -m src.train_bilstm

# Phase 5: Fine-tune DistilBERT and RoBERTa on GPU
python -m src.train_transformers

# Phase 6: Benchmark Whisper ASR, extract NER triples, & run BART summarization
python -m src.run_phase6

# Phase 7: Re-compile Academic Report, Presentation & Notebook
python -m src.generate_report
python -m src.generate_presentation
python -m src.generate_notebook
```

---

## 📄 Deliverables Summary

1. **Academic Capstone Report**: [`report/NLP_Project_Report.docx`](report/NLP_Project_Report.docx)  
   - 35+ pages adhering to IEEE formatting, containing all 15 required chapters, 14 embedded figures, 7 empirical data tables, and 18 academic references.
2. **Project Presentation**: [`presentation/NLP_Project_Presentation.pptx`](presentation/NLP_Project_Presentation.pptx)  
   - 12 widescreen (16:9) slides detailing motivation, pipeline architecture, empirical leaderboard, and demo walkthrough.
3. **Master Jupyter Notebook**: [`notebooks/NLP_Project_Full_Pipeline.ipynb`](notebooks/NLP_Project_Full_Pipeline.ipynb)  
   - Top-to-bottom runnable notebook mirroring the 15 chapters of the report.
4. **End-to-End CLI Demo**: [`demo.py`](demo.py)  
   - Production-ready script connecting Whisper, RoBERTa, spaCy, and BART.

---

## 👥 Authors & Academic Attribution

- **Student**: [STUDENT NAME] (Roll No: [ROLL NO])
- **Supervisor**: [GUIDE NAME]
- **Department**: Department of Computer Science and Engineering
- **Institution**: [INSTITUTE NAME]
- **Academic Year**: 2026-2027
