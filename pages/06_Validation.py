"""
Page for corpus upload, cleaning, translation, and processing.
"""

import streamlit as st
import pandas as pd
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from gui.styles.custom_css import apply_custom_css, styled_header
from models.llm_factory import LLMFactory
from utils.file_handlers import FileHandler
from config.settings import MODELS_CONFIG, CORPORA_DIR, MAX_CORPUS_SIZE_MB
from config.languages import get_supported_languages, STOP_WORDS

st.set_page_config(
    page_title="Corpus Manager",
    page_icon="📊",
    layout="wide"
)

apply_custom_css()

# Initialize session state
if 'corpus_text' not in st.session_state:
    st.session_state.corpus_text = ""
if 'corpus_metadata' not in st.session_state:
    st.session_state.corpus_metadata = {}
if 'cleaned_corpus' not in st.session_state:
    st.session_state.cleaned_corpus = ""

# Header
st.title("📊 Corpus Manager")
st.markdown("""
Upload, clean, translate, and process text corpora for African languages.
Use LLMs for intelligent cleaning and translation.
""")

st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    language = st.selectbox(
        "Corpus Language",
        get_supported_languages()
    )
    
    st.divider()
    
    # LLM settings for cleaning/translation
    st.subheader("LLM Settings")
    
    provider = st.selectbox(
        "Provider",
        ["OpenAI", "Claude", "Gemini"]
    )
    
    provider_key = provider.lower()
    model = st.selectbox(
        "Model",
        MODELS_CONFIG[provider_key]['models']
    )
    
    api_key = st.text_input(
        "API Key",
        type="password",
        key="corpus_api_key"
    )

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "🧹 Clean", "🌐 Translate", "📊 Statistics"])

with tab1:
    styled_header("Upload Corpus")
    
    col_up1, col_up2 = st.columns([2, 1])
    
    with col_up1:
        upload_method = st.radio(
            "Upload Method",
            ["File Upload", "Manual Entry", "URL"],
            horizontal=True
        )
    
    corpus_text = ""
    
    if upload_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Upload corpus file",
            type=['txt', 'csv', 'json'],
            help=f"Max size: {MAX_CORPUS_SIZE_MB}MB"
        )
        
        if uploaded_file:
            try:
                # Check file size
                file_size = uploaded_file.size / (1024 * 1024)  # MB
                if file_size > MAX_CORPUS_SIZE_MB:
                    st.error(f"❌ File too large: {file_size:.1f}MB (max: {MAX_CORPUS_SIZE_MB}MB)")
                else:
                    if uploaded_file.name.endswith('.txt'):
                        corpus_text = uploaded_file.read().decode('utf-8')
                    
                    elif uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        # Concatenate all text columns
                        text_cols = df.select_dtypes(include=['object']).columns
                        corpus_text = '\n'.join(df[text_cols].apply(
                            lambda x: ' '.join(x.dropna().astype(str)), axis=1
                        ))
                    
                    elif uploaded_file.name.endswith('.json'):
                        import json
                        data = json.loads(uploaded_file.read())
                        # Extract all string values
                        def extract_text(obj):
                            if isinstance(obj, str):
                                return obj
                            elif isinstance(obj, dict):
                                return ' '.join([extract_text(v) for v in obj.values()])
                            elif isinstance(obj, list):
                                return ' '.join([extract_text(item) for item in obj])
                            return ''
                        
                        corpus_text = extract_text(data)
                    
                    st.session_state.corpus_text = corpus_text
                    st.session_state.corpus_metadata = {
                        'filename': uploaded_file.name,
                        'size_mb': file_size,
                        'language': language
                    }
                    
                    st.success(f"✅ Loaded {file_size:.2f}MB of text")
            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    elif upload_method == "Manual Entry":
        corpus_text = st.text_area(
            "Enter or paste text",
            height=400,
            placeholder="Paste your corpus text here..."
        )
        
        if corpus_text:
            st.session_state.corpus_text = corpus_text
            st.session_state.corpus_metadata = {
                'source': 'manual_entry',
                'language': language
            }
    
    else:  # URL
        url = st.text_input("Enter URL")
        
        if url and st.button("Fetch from URL"):
            try:
                import requests
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                corpus_text = response.text
                
                st.session_state.corpus_text = corpus_text
                st.session_state.corpus_metadata = {
                    'source': 'url',
                    'url': url,
                    'language': language
                }
                
                st.success("✅ Fetched content from URL")
            
            except Exception as e:
                st.error(f"❌ Error fetching URL: {str(e)}")
    
    # Preview
    if st.session_state.corpus_text:
        st.divider()
        st.markdown("#### 📄 Preview")
        
        preview_length = st.slider("Preview length (characters)", 100, 5000, 1000)
        st.text_area(
            "Corpus preview",
            st.session_state.corpus_text[:preview_length],
            height=200,
            disabled=True
        )
        
        # Basic stats
        text = st.session_state.corpus_text
        col_prev1, col_prev2, col_prev3 = st.columns(3)
        
        with col_prev1:
            st.metric("Characters", len(text))
        
        with col_prev2:
            word_count = len(text.split())
            st.metric("Words", word_count)
        
        with col_prev3:
            line_count = len(text.split('\n'))
            st.metric("Lines", line_count)

with tab2:
    styled_header("Corpus Cleaning")
    
    if not st.session_state.corpus_text:
        st.warning("⚠️ Please upload a corpus first")
    else:
        st.markdown("#### Cleaning Options")
        
        col_clean1, col_clean2 = st.columns(2)
        
        with col_clean1:
            st.markdown("**Basic Cleaning:**")
            
            remove_urls = st.checkbox("Remove URLs", value=True)
            remove_emails = st.checkbox("Remove emails", value=True)
            remove_numbers = st.checkbox("Remove numbers", value=False)
            remove_punctuation = st.checkbox("Remove punctuation", value=False)
            remove_extra_whitespace = st.checkbox("Remove extra whitespace", value=True)
            lowercase = st.checkbox("Convert to lowercase", value=False)
        
        with col_clean2:
            st.markdown("**Advanced Cleaning:**")
            
            remove_stopwords = st.checkbox(
                "Remove stop words",
                value=False,
                help="Remove common words with little semantic value"
            )
            
            remove_short_words = st.checkbox("Remove short words", value=False)
            if remove_short_words:
                min_word_length = st.slider("Minimum word length", 1, 5, 2)
            
            remove_duplicates = st.checkbox("Remove duplicate lines", value=True)
            
            use_llm_cleaning = st.checkbox(
                "Use LLM for intelligent cleaning",
                value=False,
                help="Use LLM to intelligently clean and normalize text"
            )
        
        # Clean button
        if st.button("🧹 Clean Corpus", type="primary", use_container_width=True):
            cleaned_text = st.session_state.corpus_text
            
            with st.spinner("Cleaning corpus..."):
                # Basic cleaning
                if remove_urls:
                    cleaned_text = re.sub(r'http\S+|www.\S+', '', cleaned_text)
                
                if remove_emails:
                    cleaned_text = re.sub(r'\S+@\S+', '', cleaned_text)
                
                if remove_numbers:
                    cleaned_text = re.sub(r'\d+', '', cleaned_text)
                
                if remove_punctuation:
                    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
                
                if lowercase:
                    cleaned_text = cleaned_text.lower()
                
                if remove_extra_whitespace:
                    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                    cleaned_text = '\n'.join([line.strip() for line in cleaned_text.split('\n')])
                
                # Advanced cleaning
                if remove_stopwords:
                    stop_words = STOP_WORDS.get(language.lower(), [])
                    words = cleaned_text.split()
                    cleaned_text = ' '.join([w for w in words if w.lower() not in stop_words])
                
                if remove_short_words:
                    words = cleaned_text.split()
                    cleaned_text = ' '.join([w for w in words if len(w) >= min_word_length])
                
                if remove_duplicates:
                    lines = cleaned_text.split('\n')
                    seen = set()
                    unique_lines = []
                    for line in lines:
                        if line.strip() and line.strip() not in seen:
                            unique_lines.append(line)
                            seen.add(line.strip())
                    cleaned_text = '\n'.join(unique_lines)
                
                # LLM cleaning
                if use_llm_cleaning and api_key:
                    try:
                        client = LLMFactory.create_client(provider_key, api_key, model)
                        
                        # Process in chunks
                        chunk_size = 2000
                        chunks = [cleaned_text[i:i+chunk_size] 
                                 for i in range(0, len(cleaned_text), chunk_size)]
                        
                        cleaned_chunks = []
                        progress = st.progress(0)
                        
                        for idx, chunk in enumerate(chunks[:5]):  # Limit to first 5 chunks
                            prompt = f"""Clean and normalize this {language} text:
- Fix spelling errors
- Remove nonsensical text
- Preserve meaning
- Keep the same language

Text:
{chunk}

Cleaned text:"""
                            
                            cleaned_chunk = client.generate(prompt, temperature=0.3, max_tokens=2500)
                            cleaned_chunks.append(cleaned_chunk)
                            progress.progress((idx + 1) / min(5, len(chunks)))
                        
                        cleaned_text = '\n'.join(cleaned_chunks)
                        progress.empty()
                    
                    except Exception as e:
                        st.warning(f"⚠️ LLM cleaning failed: {str(e)}")
                
                # Store cleaned corpus
                st.session_state.cleaned_corpus = cleaned_text
                
                st.success("✅ Corpus cleaned successfully!")
        
        # Show cleaned corpus
        if st.session_state.cleaned_corpus:
            st.divider()
            st.markdown("#### 🧹 Cleaned Corpus")
            
            col_result1, col_result2 = st.columns([3, 1])
            
            with col_result1:
                preview_clean = st.slider("Preview length", 100, 5000, 1000, key="clean_preview")
                st.text_area(
                    "Cleaned text",
                    st.session_state.cleaned_corpus[:preview_clean],
                    height=300,
                    disabled=True
                )
            
            with col_result2:
                # Comparison stats
                original_words = len(st.session_state.corpus_text.split())
                cleaned_words = len(st.session_state.cleaned_corpus.split())
                reduction = ((original_words - cleaned_words) / original_words * 100) if original_words > 0 else 0
                
                st.metric("Original Words", original_words)
                st.metric("Cleaned Words", cleaned_words)
                st.metric("Reduction", f"{reduction:.1f}%")
            
            # Download
            st.download_button(
                "📥 Download Cleaned Corpus",
                st.session_state.cleaned_corpus,
                f"cleaned_corpus_{language}.txt",
                "text/plain",
                use_container_width=True
            )

with tab3:
    styled_header("Translation")
    
    if not st.session_state.corpus_text:
        st.warning("⚠️ Please upload a corpus first")
    elif not api_key:
        st.warning("⚠️ Please enter an API key in the sidebar")
    else:
        st.markdown("#### Translation Settings")
        
        col_trans1, col_trans2 = st.columns(2)
        
        with col_trans1:
            source_lang = st.selectbox(
                "Source Language",
                ["Auto-detect"] + get_supported_languages() + ["English"],
                help="Language of the corpus"
            )
        
        with col_trans2:
            target_lang = st.selectbox(
                "Target Language",
                get_supported_languages() + ["English"],
                help="Language to translate to"
            )
        
        translation_style = st.selectbox(
            "Translation Style",
            ["Literal", "Natural", "Formal", "Casual"],
            help="How to translate the text"
        )
        
        # Translate button
        if st.button("🌐 Translate", type="primary", use_container_width=True):
            try:
                client = LLMFactory.create_client(provider_key, api_key, model)
                
                # Get text to translate
                text_to_translate = st.session_state.cleaned_corpus if st.session_state.cleaned_corpus else st.session_state.corpus_text
                
                # Split into chunks
                chunk_size = 1500
                chunks = [text_to_translate[i:i+chunk_size] 
                         for i in range(0, len(text_to_translate), chunk_size)]
                
                translated_chunks = []
                progress_bar = st.progress(0)
                status = st.empty()
                
                for idx, chunk in enumerate(chunks[:10]):  # Limit to 10 chunks
                    status.info(f"Translating chunk {idx + 1}/{min(10, len(chunks))}...")
                    
                    prompt = f"""Translate this text from {source_lang} to {target_lang}.
Style: {translation_style}

Text:
{chunk}

Translation:"""
                    
                    translation = client.generate(prompt, temperature=0.3, max_tokens=2000)
                    translated_chunks.append(translation)
                    
                    progress_bar.progress((idx + 1) / min(10, len(chunks)))
                
                full_translation = '\n'.join(translated_chunks)
                
                progress_bar.empty()
                status.success("✅ Translation complete!")
                
                # Display
                st.text_area(
                    f"Translation ({target_lang})",
                    full_translation,
                    height=400
                )
                
                # Download
                st.download_button(
                    "📥 Download Translation",
                    full_translation,
                    f"translation_{source_lang}_to_{target_lang}.txt",
                    "text/plain"
                )
            
            except Exception as e:
                st.error(f"❌ Translation error: {str(e)}")

with tab4:
    styled_header("Corpus Statistics")
    
    if not st.session_state.corpus_text:
        st.info("Upload a corpus to see statistics")
    else:
        text = st.session_state.cleaned_corpus if st.session_state.cleaned_corpus else st.session_state.corpus_text
        
        # Calculate statistics
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        unique_words = len(set([w.lower() for w in words]))
        lines = text.split('\n')
        line_count = len([l for l in lines if l.strip()])
        
        # Display metrics
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Characters", f"{char_count:,}")
        
        with col_stat2:
            st.metric("Words", f"{word_count:,}")
        
        with col_stat3:
            st.metric("Unique Words", f"{unique_words:,}")
        
        with col_stat4:
            diversity = (unique_words / word_count * 100) if word_count > 0 else 0
            st.metric("Lexical Diversity", f"{diversity:.1f}%")
        
        st.divider()
        
        # Word frequency
        st.markdown("#### 📊 Top Words")
        
        from collections import Counter
        word_freq = Counter([w.lower() for w in words if len(w) > 2])
        top_words = word_freq.most_common(20)
        
        df_freq = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
        
        col_freq1, col_freq2 = st.columns([1, 1])
        
        with col_freq1:
            st.dataframe(df_freq, use_container_width=True, height=400)
        
        with col_freq2:
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[
                go.Bar(x=df_freq['Word'], y=df_freq['Frequency'])
            ])
            
            fig.update_layout(
                title="Top 20 Words",
                xaxis_title="Word",
                yaxis_title="Frequency",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>💡 <strong>Tip:</strong> Clean your corpus before translation for better results.</p>
</div>
""", unsafe_allow_html=True)