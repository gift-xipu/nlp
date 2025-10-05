"""
Page for generating sentiment word lists using LLMs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import json
from datetime import datetime

from gui.styles.custom_css import apply_custom_css, styled_header
from models.llm_factory import LLMFactory
from tasks.list_words import ListWordsTask
from utils.file_handlers import FileHandler
from config.settings import MODELS_CONFIG, SENTIMENT_CATEGORIES, PROMPT_STRATEGIES, EXPORT_FORMATS, EXPORTS_DIR
from config.languages import get_supported_languages

# Page config
st.set_page_config(
    page_title="List Words - African Languages Sentiment",
    page_icon="📝",
    layout="wide"
)

apply_custom_css()

# Initialize session state
if 'generated_words' not in st.session_state:
    st.session_state.generated_words = []
if 'generation_stats' not in st.session_state:
    st.session_state.generation_stats = {}

# Header
st.title("📝 Word List Generation")
st.markdown("""
Generate sentiment-bearing words in African languages using LLMs.
Target: **1000 unique words** per sentiment category.
""")

st.divider()

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # LLM Selection
    st.subheader("Model Settings")
    provider = st.selectbox(
        "LLM Provider",
        ["OpenAI", "Claude", "Gemini"],
        help="Select the language model provider"
    )
    
    provider_key = provider.lower()
    available_models = MODELS_CONFIG[provider_key]['models']
    default_model = MODELS_CONFIG[provider_key]['default']
    
    model = st.selectbox(
        "Model",
        available_models,
        index=available_models.index(default_model) if default_model in available_models else 0
    )
    
    api_key = st.text_input(
        "API Key",
        type="password",
        help="Enter your API key",
        key="api_key_input"
    )
    
    st.divider()
    
    # Generation Parameters
    st.subheader("Generation Parameters")
    
    language = st.selectbox(
        "Language",
        get_supported_languages(),
        help="Target African language"
    )
    
    sentiment = st.selectbox(
        "Sentiment",
        SENTIMENT_CATEGORIES,
        help="Type of sentiment words to generate"
    )
    
    prompt_strategy = st.selectbox(
        "Prompt Strategy",
        PROMPT_STRATEGIES,
        index=1,  # Default to few-shot
        help="Prompting technique"
    )
    
    st.divider()
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        target_count = st.number_input(
            "Target Word Count",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Number of unique words to generate"
        )
        
        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.7, 0.1,
            help="Higher = more creative"
        )
        
        max_tokens = st.number_input(
            "Max Tokens per Batch",
            100, 1000, 500, 50,
            help="Maximum tokens per API call"
        )
        
        enable_validation = st.checkbox(
            "Enable Word Validation",
            value=True,
            help="Validate words against linguistic rules"
        )
    
    st.divider()
    
    # Export Settings
    st.subheader("📥 Export Settings")
    export_formats = st.multiselect(
        "Export Formats",
        EXPORT_FORMATS,
        default=['csv', 'txt'],
        help="Select export formats"
    )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    styled_header("Generation Controls")
    
    # Start generation button
    if st.button("🚀 Start Generation", type="primary", use_container_width=True):
        if not api_key:
            st.error("❌ Please enter an API key in the sidebar")
        else:
            try:
                # Initialize LLM client
                with st.spinner(f"Initializing {provider} client..."):
                    client = LLMFactory.create_client(
                        provider=provider_key,
                        api_key=api_key,
                        model=model
                    )
                
                # Initialize task
                task = ListWordsTask(
                    llm_client=client,
                    language=language,
                    sentiment=sentiment,
                    prompt_strategy=prompt_strategy
                )
                
                # Progress containers
                progress_bar = st.progress(0)
                status_text = st.empty()
                stats_container = st.empty()
                
                def update_progress(current, total, batch):
                    progress = current / total
                    progress_bar.progress(min(progress, 1.0))
                    status_text.info(
                        f"📊 Generated {current}/{total} words (Batch {batch})"
                    )
                
                # Generate words
                st.info(f"🌱 Starting generation of {target_count} {sentiment} words in {language}...")
                
                words, stats = task.generate_words(
                    target_count=target_count,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    progress_callback=update_progress,
                    validate=enable_validation
                )
                
                # Store in session state
                st.session_state.generated_words = words
                st.session_state.generation_stats = stats
                
                # Clear progress
                progress_bar.progress(1.0)
                status_text.success(
                    f"✅ Successfully generated {len(words)} unique words!"
                )
                
                # Display stats
                with stats_container:
                    st.metric("Total Generated", stats['total_generated'])
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Unique Words", stats['unique_words'])
                    with col_b:
                        st.metric("Duplicates Removed", stats['duplicates_removed'])
                    with col_c:
                        st.metric("Invalid Removed", stats['invalid_removed'])
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

with col2:
    styled_header("Status")
    
    if st.session_state.generated_words:
        st.success(f"✅ {len(st.session_state.generated_words)} words ready")
        
        if st.session_state.generation_stats:
            stats = st.session_state.generation_stats
            st.metric("Batches Processed", stats.get('batches_processed', 0))
            
            efficiency = (stats['unique_words'] / stats['total_generated'] * 100) if stats['total_generated'] > 0 else 0
            st.metric("Efficiency", f"{efficiency:.1f}%")
    else:
        st.info("No words generated yet")

# Results section
if st.session_state.generated_words:
    st.divider()
    styled_header("Generated Words")
    
    words = st.session_state.generated_words
    
    # Display options
    col_disp1, col_disp2 = st.columns([3, 1])
    
    with col_disp1:
        display_count = st.slider(
            "Words to display",
            10, min(len(words), 200), min(50, len(words)),
            help="Number of words to show in preview"
        )
    
    with col_disp2:
        sort_option = st.selectbox(
            "Sort by",
            ["Original Order", "Alphabetical", "Reverse Alphabetical"]
        )
    
    # Sort words
    display_words = words.copy()
    if sort_option == "Alphabetical":
        display_words.sort(key=lambda x: x.get('word', ''))
    elif sort_option == "Reverse Alphabetical":
        display_words.sort(key=lambda x: x.get('word', ''), reverse=True)
    
    # Display as text area for easy copying
    words_text = '\n'.join([
        f"{w.get('word', '')}: {w.get('translation', '')}"
        for w in display_words[:display_count]
    ])
    
    st.text_area(
        f"First {display_count} words",
        words_text,
        height=400,
        help="Preview of generated words"
    )
    
    # Export section
    st.divider()
    styled_header("Export Options")
    
    col_exp1, col_exp2 = st.columns([2, 1])
    
    with col_exp1:
        include_metadata = st.checkbox(
            "Include metadata",
            value=True,
            help="Include generation metadata in exports"
        )
    
    with col_exp2:
        if st.button("📥 Export All Formats", use_container_width=True):
            try:
                metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'language': language,
                    'sentiment': sentiment,
                    'prompt_strategy': prompt_strategy,
                    'model': model,
                    'provider': provider,
                    'total_words': len(words),
                    'statistics': st.session_state.generation_stats
                } if include_metadata else {}
                
                exported_files = []
                
                for fmt in export_formats:
                    # Smart filename: groups by language, LLM, and prompt type
                    filename = FileHandler.generate_filename(
                        'lexicon',
                        language,
                        llm=provider_key,
                        prompt_type=prompt_strategy,
                        extension=fmt,
                        include_timestamp=False  # No timestamp for grouping
                    )
                    filepath = EXPORTS_DIR / filename
                    
                    # Append to existing file (groups all sentiments together)
                    FileHandler.append_to_file(filepath, words, format=fmt)
                    
                    exported_files.append(str(filepath))
                
                st.success(f"✅ Exported to {len(exported_files)} formats!")
                st.info(f"💡 Words grouped by: {language} + {provider} + {prompt_strategy}")
                
                for file in exported_files:
                    file_path = Path(file)
                    # Count total words in file
                    if file_path.suffix == '.csv':
                        total_in_file = len(pd.read_csv(file_path))
                        st.text(f"📄 {file_path.name} ({total_in_file} total words)")
                    else:
                        st.text(f"📄 {file_path.name}")
                
            except Exception as e:
                st.error(f"❌ Export error: {str(e)}")
    
    # Download buttons
    st.markdown("#### Quick Downloads")
    
    download_cols = st.columns(len(export_formats) if export_formats else 3)
    
    for idx, fmt in enumerate(export_formats if export_formats else ['csv', 'txt', 'json']):
        with download_cols[idx]:
            if fmt == 'csv':
                df = pd.DataFrame(words)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    f"📄 CSV",
                    csv_data,
                    f"lexicon_{language}_{sentiment}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            elif fmt == 'txt':
                txt_data = '\n'.join([
                    f"{w.get('word', '')}: {w.get('translation', '')}"
                    for w in words
                ])
                st.download_button(
                    f"📄 TXT",
                    txt_data,
                    f"lexicon_{language}_{sentiment}.txt",
                    "text/plain",
                    use_container_width=True
                )
            
            elif fmt == 'json':
                json_data = json.dumps(words, indent=2, ensure_ascii=False)
                st.download_button(
                    f"📄 JSON",
                    json_data,
                    f"lexicon_{language}_{sentiment}.json",
                    "application/json",
                    use_container_width=True
                )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>💡 <strong>Tip:</strong> Use few-shot prompting for better quality. Generate 1000+ words for robust coverage.</p>
</div>
""", unsafe_allow_html=True)