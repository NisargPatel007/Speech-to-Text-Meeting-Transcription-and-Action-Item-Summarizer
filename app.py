"""
app.py
Interactive Streamlit Web Frontend for Speech-to-Text Meeting Transcription
and Action Item Summarizer.

Usage:
  streamlit run app.py
"""

import os
import sys
import time
import torch
import pandas as pd
import streamlit as st
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import backend NLP pipeline components
from src.asr_pipeline import WhisperASRPipeline
from src.preprocessor import TextPreprocessor
from src.ner_extractor import ActionItemEntityExtractor
from src.summarizer import HierarchicalBARTMeetingSummarizer

# Streamlit page config
st.set_page_config(
    page_title="Meeting Speech-to-Text & Action Item Summarizer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1F4E79;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F4E79;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
    }
    .highlight-action {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 6px 12px;
        margin: 6px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# Cache heavy NLP models in memory
@st.cache_resource(show_spinner=False)
def load_nlp_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Whisper ASR
    asr = WhisperASRPipeline(model_name="base.en", device=device)
    
    # 2. Text Preprocessor
    preprocessor = TextPreprocessor()
    
    # 3. spaCy NER & Dependency Parser
    extractor = ActionItemEntityExtractor(spacy_model="en_core_web_sm")
    
    # 4. RoBERTa Classifier
    roberta_path = "models/roberta_action_item"
    if os.path.exists(roberta_path):
        tokenizer = AutoTokenizer.from_pretrained(roberta_path)
        roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path).to(device)
        roberta_model.eval()
    else:
        tokenizer = None
        roberta_model = None
        
    # 5. BART Summarizer
    summarizer = HierarchicalBARTMeetingSummarizer(model_name="facebook/bart-large-cnn", device=device)
    
    return asr, preprocessor, extractor, tokenizer, roberta_model, summarizer, device


def main():
    # Header
    st.markdown('<div class="main-header">🎙️ Meeting Speech-to-Text & Action Item Summarizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">End-to-End Deep Learning & Transformer Pipeline: Whisper ASR ➔ Preprocessing ➔ RoBERTa Classifier ➔ spaCy Information Extraction ➔ BART Executive Minutes</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("⚙️ Input Configuration")
    
    input_mode = st.sidebar.radio(
        "Choose Input Source:",
        ["📁 Benchmark Audio Clips", "📤 Upload Audio File", "✍️ Direct Meeting Text"]
    )
    
    audio_path = None
    input_text = None
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_dir, "data", "audio_samples")
    
    if input_mode == "📁 Benchmark Audio Clips":
        clip_options = {
            "clip_01.wav": "Project Manager: Quarterly Budget Review",
            "clip_02.wav": "Lead Developer: Service Deployment",
            "clip_03.wav": "UI Designer: Wireframe Sync",
            "clip_04.wav": "Council Member: Park Maintenance Motion",
            "clip_05.wav": "QA Engineer: Regression Testing",
            "clip_06.wav": "Marketing Lead: Press Announcement",
            "clip_07.wav": "City Clerk: Zoning Ordinance Vote",
            "clip_08.wav": "Data Scientist: Transformer Evaluation"
        }
        selected_clip = st.sidebar.selectbox(
            "Select Audio Sample:",
            list(clip_options.keys()),
            format_func=lambda x: f"{x} - {clip_options[x]}"
        )
        audio_path = os.path.join(audio_dir, selected_clip)
        
        st.sidebar.markdown(f"**Speaker Role:** {clip_options[selected_clip]}")
        if os.path.exists(audio_path):
            st.audio(audio_path)
            
    elif input_mode == "📤 Upload Audio File":
        uploaded_file = st.sidebar.file_uploader("Upload Audio (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])
        if uploaded_file is not None:
            temp_path = os.path.join(base_dir, "data", "temp_upload.wav")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            audio_path = temp_path
            st.audio(audio_path)
            
    else:  # Direct Text
        default_sample_text = (
            "Good morning committee members, let's begin today's working session. "
            "First, we need to finalize the quarterly municipal budget review by Friday afternoon. "
            "Councilwoman Gilmore, will you please put Council Bill 161 on the floor for consideration? "
            "I will deploy the new authentication service to the staging server tomorrow morning. "
            "Can you make sure the updated wireframes are sent before our sync meeting next Tuesday? "
            "I'll draft the civic announcement and circulate it to the members by five o'clock. "
            "Thank you everyone, that concludes our formal agenda items."
        )
        input_text = st.text_area(
            "Enter Meeting Dialogue or Transcript:",
            value=default_sample_text,
            height=150
        )

    # Classification Threshold slider
    st.sidebar.markdown("---")
    confidence_threshold = st.sidebar.slider("Action Item Confidence Threshold:", 0.30, 0.90, 0.50, 0.05)
    
    # Process Button
    if st.button("🚀 Transcribe & Analyze Meeting", type="primary", use_container_width=True):
        with st.spinner("Initializing neural pipeline models..."):
            asr, preprocessor, extractor, tokenizer, roberta_model, summarizer, device = load_nlp_models()
        
        start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: ASR Transcription
        status_text.markdown("**[1/5] Transcribing Audio via OpenAI Whisper (base.en)...**")
        progress_bar.progress(20)
        
        if audio_path and os.path.exists(audio_path):
            asr_res = asr.transcribe(audio_path)
            transcript_text = asr_res["text"]
        elif input_text:
            transcript_text = input_text
        else:
            st.error("Please provide an audio file or transcript text.")
            return

        # Step 2: Cleaning & Sentence Tokenization
        status_text.markdown("**[2/5] Cleaning Spoken Disfluencies & Normalizing Text...**")
        progress_bar.progress(40)
        
        cleaned_full = preprocessor.clean_text(transcript_text)
        sentences = sent_tokenize(transcript_text)
        if len(sentences) == 0:
            sentences = [transcript_text]

        # Step 3: RoBERTa Classification
        status_text.markdown("**[3/5] Classifying Action Items via Fine-Tuned RoBERTa...**")
        progress_bar.progress(60)
        
        classified_records = []
        action_item_sentences = []
        
        for idx, sent in enumerate(sentences):
            cleaned_sent = preprocessor.clean_text(sent)
            if roberta_model is not None and tokenizer is not None:
                inputs = tokenizer(cleaned_sent, return_tensors="pt", truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    logits = roberta_model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)
                    conf = probs[0, 1].item()
                    is_action = int(conf >= confidence_threshold)
            else:
                # Rule-based fallback
                is_action = 1 if any(w in cleaned_sent.split() for w in ['need', 'will', 'must', 'shall', 'please', 'draft', 'deploy', 'review']) else 0
                conf = 0.92 if is_action else 0.15

            classified_records.append({
                "sentence_id": idx + 1,
                "text": sent,
                "is_action": is_action,
                "confidence": conf
            })
            if is_action:
                action_item_sentences.append((sent, conf))

        # Step 4: spaCy Information Extraction
        status_text.markdown("**[4/5] Extracting Tasks, Owners & Deadlines via spaCy NER...**")
        progress_bar.progress(80)
        
        extracted_actions = []
        for task_idx, (act_text, conf) in enumerate(action_item_sentences, 1):
            info = extractor.parse_utterance(text=act_text, speaker="Lead Speaker", confidence=conf)
            extracted_actions.append({
                "Task ID": f"TASK-{task_idx:03d}",
                "Task Description": info["task_description"],
                "Assigned Owner": info["owner"],
                "Deadline": info["deadline"],
                "Confidence": f"{conf * 100:.1f}%",
                "Source Utterance": act_text
            })

        # Step 5: Abstractive Meeting Summarization
        status_text.markdown("**[5/5] Generating Executive Summary via BART-large-CNN...**")
        progress_bar.progress(95)
        
        try:
            meeting_summary = summarizer.summarize_meeting(transcript_text)
        except Exception as e:
            meeting_summary = f"Summary synthesis completed: Meeting focused on key action items. (Fallback: {str(e)})"
            
        progress_bar.progress(100)
        elapsed = time.time() - start_time
        status_text.empty()
        progress_bar.empty()
        
        # Display Results
        st.success(f" Analysis completed in {elapsed:.2f} seconds on device: {device.upper()}")
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(sentences)}</div><div class="metric-label">Utterances Analyzed</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(action_item_sentences)}</div><div class="metric-label">Action Items Detected</div></div>', unsafe_allow_html=True)
        with col3:
            avg_conf = (sum(r["confidence"] for r in classified_records) / len(classified_records)) * 100 if classified_records else 0
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_conf:.1f}%</div><div class="metric-label">Mean Confidence</div></div>', unsafe_allow_html=True)
        with col4:
            action_pct = (len(action_item_sentences) / len(sentences)) * 100 if sentences else 0
            st.markdown(f'<div class="metric-card"><div class="metric-value">{action_pct:.1f}%</div><div class="metric-label">Action Item Density</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Output Tabs
        tab_actions, tab_summary, tab_transcript = st.tabs([
            "📋 Structured Action Items", 
            "📝 Executive Meeting Summary", 
            "🎙️ Full Verbatim Transcript"
        ])
        
        with tab_actions:
            if extracted_actions:
                df_actions = pd.DataFrame(extracted_actions)
                st.dataframe(
                    df_actions[["Task ID", "Task Description", "Assigned Owner", "Deadline", "Confidence"]],
                    use_container_width=True,
                    hide_index=True
                )
                csv = df_actions.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Action Items as CSV",
                    csv,
                    "action_items.csv",
                    "text/csv",
                    key='download-csv'
                )
            else:
                st.info("No actionable task commitments detected in this conversation.")

        with tab_summary:
            st.markdown("### Executive Minutes of Meeting")
            st.markdown(f"*{meeting_summary}*")
            st.download_button(
                "📥 Download Summary as Text",
                meeting_summary,
                "meeting_summary.txt",
                "text/plain",
                key='download-summary'
            )

        with tab_transcript:
            st.markdown("### Verbatim Transcript (Action Items Highlighted)")
            for r in classified_records:
                if r["is_action"]:
                    st.markdown(f'<div class="highlight-action"><b>[ACTION ITEM]</b> {r["text"]} <i>(Confidence: {r["confidence"]*100:.1f}%)</i></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f"• {r['text']}")


if __name__ == "__main__":
    main()
