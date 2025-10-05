"""
Page for labeling words with sentiment, scores, and reasoning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import json

from gui.styles.custom_css import apply_custom_css, styled_header
from models.llm_factory import LLMFactory
from tasks.sentiment_bearing import SentimentBearingTask
from utils.file_handlers import FileHandler
from config.settings import MODELS_CONFIG
from config.languages import get_supported_languages

st.set_page_config(
    page_title="Sentiment Bearing",
    page_icon="🎯",
    layout="wide"
)

apply_custom_css()

if 'labeled_words' not in st.session_state:
    st.session_state.labeled_words = []
if 'validation_stats' not in st.session_state:
    st.session_state.validation_stats = {}

st.title("🎯 Sentiment Bearing Analysis")
st.markdown("""
Label words with **sentiment**, **confidence scores**, and **reasoning**.
Upload a word list or use previously generated words.
""")

st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Model Settings")
    provider = st.selectbox("LLM Provider", ["OpenAI", "Claude", "Gemini"])
    
    provider_key = provider.lower()
    model = st.selectbox("Model", MODELS_CONFIG[provider_key]['models'])
    
    api_key = st.text_input("API Key", type="password", key="sentiment_api_key")
    
    st.divider()
    
    st.subheader("Analysis Settings")
    language = st.selectbox("Language", get_supported_languages())
    prompt_strategy = st.selectbox("Prompt Strategy", ["zero-shot", "few-shot", "in-context"], index=1)
    
    with st.expander("🔧 Advanced"):
        batch_size = st.slider("Batch Size", 1, 20, 10)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload Words", "🔍 Analyze", "📊 Results"])

with tab1:
    styled_header("Input Word List")
    
    input_method = st.radio("Input Method", ["Use Generated Words", "Upload File", "Manual Entry"], horizontal=True)
    
    words_to_analyze = []
    
    if input_method == "Use Generated Words":
        if 'generated_words' in st.session_state and st.session_state.generated_words:
            words_to_analyze = st.session_state.generated_words
            st.success(f"✅ Loaded {len(words_to_analyze)} words from previous generation")
        else:
            st.warning("⚠️ No generated words found. Generate words first or upload a file.")
    
    elif input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload word list", type=['csv', 'txt', 'json'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    words_to_analyze = df.to_dict('records')
                elif uploaded_file.name.endswith('.txt'):
                    content = uploaded_file.read().decode('utf-8')
                    for line in content.split('\n'):
                        if ':' in line:
                            parts = line.split(':', 1)
                            words_to_analyze.append({
                                'word': parts[0].strip(),
                                'translation': parts[1].strip()
                            })
                elif uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read().decode('utf-8')
                    words_to_analyze = json.loads(content)
                
                st.success(f"✅ Loaded {len(words_to_analyze)} words")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    else:  # Manual Entry
        manual_input = st.text_area("Word List", height=200, 
                                    placeholder="thabo: joy\nkatlego: success")
        if manual_input:
            for line in manual_input.split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    words_to_analyze.append({
                        'word': parts[0].strip(),
                        'translation': parts[1].strip()
                    })
            if words_to_analyze:
                st.info(f"📝 Parsed {len(words_to_analyze)} words")
    
    if words_to_analyze:
        st.session_state.words_to_analyze = words_to_analyze
        st.divider()
        st.markdown("#### Preview")
        st.dataframe(pd.DataFrame(words_to_analyze[:10]), use_container_width=True)

with tab2:
    styled_header("Sentiment Analysis")
    
    if not st.session_state.get('words_to_analyze'):
        st.warning("⚠️ Please load words in the Upload tab first")
    else:
        words = st.session_state.words_to_analyze
        st.info(f"Ready to analyze {len(words)} words")
        
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ Please enter an API key")
            else:
                try:
                    with st.spinner("Initializing..."):
                        client = LLMFactory.create_client(provider_key, api_key, model)
                    
                    task = SentimentBearingTask(client, language, prompt_strategy)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(current, total):
                        progress_bar.progress(current / total)
                        status_text.info(f"🔍 Analyzed {current}/{total} words...")
                    
                    st.info("🌱 Starting analysis...")
                    
                    labeled = task.analyze_batch(words, batch_size, temperature, update_progress)
                    stats = task.validate_sentiment_consistency(labeled)
                    
                    st.session_state.labeled_words = labeled
                    st.session_state.validation_stats = stats
                    
                    progress_bar.progress(1.0)
                    status_text.success(f"✅ Analyzed {len(labeled)} words!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with tab3:
    styled_header("Analysis Results")
    
    if not st.session_state.labeled_words:
        st.info("No results yet. Analyze words first.")
    else:
        labeled_words = st.session_state.labeled_words
        stats = st.session_state.validation_stats
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", stats.get('total_words', 0))
        with col2:
            st.metric("Avg Confidence", f"{stats.get('average_confidence', 0):.2f}")
        with col3:
            st.metric("High Confidence", stats.get('high_confidence_count', 0))
        with col4:
            st.metric("High Conf %", f"{stats.get('high_confidence_percentage', 0):.1f}%")
        
        # Distribution
        st.divider()
        dist = stats.get('sentiment_distribution', {})
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("😊 Positive", dist.get('positive', 0))
        with col_d2:
            st.metric("😔 Negative", dist.get('negative', 0))
        with col_d3:
            st.metric("😐 Neutral", dist.get('neutral', 0))
        
        # Table
        st.divider()
        df = pd.DataFrame(labeled_words)
        st.dataframe(df, use_container_width=True, height=400)
        
        # Export
        st.divider()
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.download_button("📄 CSV", df.to_csv(index=False), 
                             f"sentiment_{language}.csv", "text/csv", use_container_width=True)
        with col_e2:
            st.download_button("📄 JSON", json.dumps(labeled_words, indent=2, ensure_ascii=False),
                             f"sentiment_{language}.json", "application/json", use_container_width=True)
        with col_e3:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Analysis', index=False)
                pd.DataFrame([stats]).to_excel(writer, sheet_name='Stats', index=False)
            st.download_button("📄 Excel", output.getvalue(),
                             f"sentiment_{language}.xlsx", use_container_width=True)