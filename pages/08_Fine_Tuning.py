"""
Create: pages/08_Fine_Tuning.py

New page in the Streamlit app for fine-tuning management
"""

import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui.styles.custom_css import apply_custom_css, styled_header
from tasks.fine_tuning import CorpusPreparator, FineTuningManager, OpenAIFineTuner
from config.languages import get_supported_languages
from config.settings import MODELS_CONFIG
import pandas as pd

st.set_page_config(
    page_title="Fine-Tuning",
    page_icon="🎓",
    layout="wide"
)

apply_custom_css()

# Initialize session state
if 'finetuning_status' not in st.session_state:
    st.session_state.finetuning_status = {}
if 'training_data_prepared' not in st.session_state:
    st.session_state.training_data_prepared = False

# Initialize manager
manager = FineTuningManager()

# Header
st.title("🎓 Fine-Tuning Management")
st.markdown("""
Fine-tune LLMs on your parallel corpora to improve performance on African languages.
""")

st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    language = st.selectbox(
        "Language",
        get_supported_languages()
    )
    
    provider = st.selectbox(
        "Provider",
        ["OpenAI", "Claude", "Gemini"],
        help="OpenAI fine-tuning available. Claude/Gemini coming soon."
    )
    
    provider_key = provider.lower()
    
    st.divider()
    
    if provider_key == 'openai':
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            key="ft_api_key"
        )
        
        model = st.selectbox(
            "Base Model",
            MODELS_CONFIG['openai']['models']
        )
    
    else:
        st.info(f"ℹ️ {provider} fine-tuning not yet available via API")
        api_key = None
        model = None

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Check Corpus",
    "🔧 Prepare Data",
    "🚀 Fine-Tune",
    "📦 My Models"
])

# TAB 1: Check Corpus
with tab1:
    styled_header("Check Your Parallel Corpus")
    
    st.markdown("""
    Before fine-tuning, verify your parallel corpus is properly formatted.
    
    **Expected structure:**
    ```
    data/translated/
    └── sepedi/          (or sesotho, setswana)
        ├── english.txt  ← One sentence per line
        └── sepedi.txt   ← Aligned with English
    ```
    """)
    
    if st.button("🔍 Check Corpus", type="primary", use_container_width=True):
        try:
            prep = CorpusPreparator('data/translated')
            
            with st.spinner(f"Loading {language} corpus..."):
                english_texts, target_texts = prep.load_parallel_corpus(language)
            
            st.success(f"✅ Found {len(english_texts)} aligned sentence pairs!")
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Pairs", len(english_texts))
            
            with col2:
                avg_en_len = sum(len(t) for t in english_texts) / len(english_texts)
                st.metric("Avg English Length", f"{avg_en_len:.0f} chars")
            
            with col3:
                avg_tgt_len = sum(len(t) for t in target_texts) / len(target_texts)
                st.metric(f"Avg {language} Length", f"{avg_tgt_len:.0f} chars")
            
            # Show sample pairs
            st.divider()
            st.markdown("### 📄 Sample Pairs")
            
            sample_size = min(10, len(english_texts))
            
            for i in range(sample_size):
                with st.expander(f"Pair {i+1}"):
                    col_en, col_tgt = st.columns(2)
                    
                    with col_en:
                        st.markdown("**English:**")
                        st.text(english_texts[i])
                    
                    with col_tgt:
                        st.markdown(f"**{language}:**")
                        st.text(target_texts[i])
            
            # Store in session state
            st.session_state.corpus_loaded = True
            st.session_state.english_texts = english_texts
            st.session_state.target_texts = target_texts
            st.session_state.corpus_language = language
            
        except FileNotFoundError as e:
            st.error(f"❌ Corpus not found: {e}")
            
            st.info("""
            **To add your corpus:**
            
            1. Create folder: `data/translated/sepedi/` (or sesotho/setswana)
            2. Add `english.txt` - one sentence per line
            3. Add `sepedi.txt` - aligned sentences
            4. Make sure line 1 in English = line 1 in Sepedi
            """)
        
        except Exception as e:
            st.error(f"❌ Error: {e}")

# TAB 2: Prepare Training Data
with tab2:
    styled_header("Prepare Training Data")
    
    if not st.session_state.get('corpus_loaded'):
        st.warning("⚠️ Please check your corpus first in the 'Check Corpus' tab")
    
    else:
        st.success(f"✅ Corpus loaded: {len(st.session_state.english_texts)} pairs")
        
        st.markdown("### Select Training Tasks")
        
        col_task1, col_task2, col_task3 = st.columns(3)
        
        with col_task1:
            include_list_words = st.checkbox(
                "📝 Word Generation",
                value=True,
                help="Train model to generate sentiment words"
            )
        
        with col_task2:
            include_sentiment = st.checkbox(
                "🎯 Sentiment Analysis",
                value=True,
                help="Train model to analyze sentiment"
            )
        
        with col_task3:
            include_translation = st.checkbox(
                "🌐 Translation",
                value=True,
                help="Train model to translate"
            )
        
        st.divider()
        
        # Load lexicon option
        st.markdown("### Optional: Load Existing Lexicon")
        st.markdown("If you have a generated lexicon, load it to improve training data quality")
        
        uploaded_lexicon = st.file_uploader(
            "Upload Lexicon (CSV/JSON)",
            type=['csv', 'json']
        )
        
        lexicon_data = None
        if uploaded_lexicon:
            try:
                if uploaded_lexicon.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_lexicon)
                    lexicon_data = df.to_dict('records')
                else:
                    import json
                    lexicon_data = json.loads(uploaded_lexicon.read())
                
                st.success(f"✅ Loaded {len(lexicon_data)} words from lexicon")
            except Exception as e:
                st.error(f"❌ Error loading lexicon: {e}")
        
        st.divider()
        
        if st.button("🔧 Prepare Training Data", type="primary", use_container_width=True):
            try:
                prep = CorpusPreparator('data/translated')
                
                with st.spinner("Preparing training data..."):
                    # Get corpus from session state
                    english_texts = st.session_state.english_texts
                    target_texts = st.session_state.target_texts
                    corpus_language = st.session_state.corpus_language
                    
                    # Create training data
                    all_data = prep.create_training_data_for_all_tasks(
                        language=corpus_language,
                        english_texts=english_texts,
                        target_texts=target_texts,
                        sentiment_lexicon=lexicon_data
                    )
                    
                    # Filter by selected tasks
                    selected_data = []
                    if include_list_words:
                        selected_data.extend(all_data['list_words'])
                    if include_sentiment:
                        selected_data.extend(all_data['sentiment_bearing'])
                    if include_translation:
                        selected_data.extend(all_data['translation'])
                    
                    if not selected_data:
                        st.error("❌ No tasks selected!")
                        st.stop()
                    
                    # Split train/val
                    train_data, val_data = prep.split_train_val(selected_data, val_ratio=0.1)
                    
                    # Save files
                    output_dir = Path('data/finetuning') / corpus_language.lower()
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    train_file = output_dir / 'train.jsonl'
                    val_file = output_dir / 'val.jsonl'
                    
                    prep.save_training_data(train_data, train_file)
                    prep.save_training_data(val_data, val_file)
                    
                    # Store in session state
                    st.session_state.train_file = str(train_file)
                    st.session_state.val_file = str(val_file)
                    st.session_state.training_data_prepared = True
                    st.session_state.prepared_language = corpus_language
                
                st.success("✅ Training data prepared!")
                
                # Show stats
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    st.metric("Training Examples", len(train_data))
                
                with col_stat2:
                    st.metric("Validation Examples", len(val_data))
                
                with col_stat3:
                    total = len(train_data) + len(val_data)
                    st.metric("Total Examples", total)
                
                st.info(f"""
                📁 **Files saved:**
                - Training: `{train_file}`
                - Validation: `{val_file}`
                """)
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# TAB 3: Fine-Tune
with tab3:
    styled_header("Fine-Tune Model")
    
    if not st.session_state.get('training_data_prepared'):
        st.warning("⚠️ Please prepare training data first in the 'Prepare Data' tab")
    
    else:
        st.success(f"✅ Training data ready for {st.session_state.prepared_language}")
        
        if provider_key != 'openai':
            st.error(f"❌ {provider} fine-tuning not yet supported via API")
            st.info("Currently only OpenAI fine-tuning is available.")
        
        elif not api_key:
            st.warning("⚠️ Please enter OpenAI API key in the sidebar")
        
        else:
            st.markdown("### Fine-Tuning Configuration")
            
            suffix = st.text_input(
                "Model Suffix",
                value=f"{language.lower()}-sentiment",
                help="Custom suffix for your fine-tuned model"
            )
            
            st.markdown("### Start Fine-Tuning")
            
            col_ft1, col_ft2 = st.columns([2, 1])
            
            with col_ft1:
                st.info(f"""
                **Ready to fine-tune:**
                - Language: {st.session_state.prepared_language}
                - Base Model: {model}
                - Training file: `{Path(st.session_state.train_file).name}`
                - Validation file: `{Path(st.session_state.val_file).name}`
                """)
            
            with col_ft2:
                if st.button("🚀 Start Fine-Tuning", type="primary", use_container_width=True):
                    try:
                        finetuner = OpenAIFineTuner(api_key)
                        
                        # Upload files
                        with st.spinner("Uploading training file..."):
                            train_file_id = finetuner.upload_file(st.session_state.train_file)
                        
                        with st.spinner("Uploading validation file..."):
                            val_file_id = finetuner.upload_file(st.session_state.val_file)
                        
                        # Start fine-tuning
                        with st.spinner("Starting fine-tuning job..."):
                            job_id = finetuner.start_finetuning(
                                train_file_id=train_file_id,
                                val_file_id=val_file_id,
                                model=model,
                                suffix=suffix
                            )
                        
                        st.success(f"✅ Fine-tuning started!")
                        
                        st.info(f"""
                        **Job ID:** `{job_id}`
                        
                        Track progress at:
                        https://platform.openai.com/finetune/{job_id}
                        """)
                        
                        # Store job info
                        st.session_state.finetuning_status[job_id] = {
                            'language': st.session_state.prepared_language,
                            'model': model,
                            'status': 'running',
                            'job_id': job_id
                        }
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Show active jobs
            if st.session_state.finetuning_status:
                st.divider()
                st.markdown("### 📊 Active Jobs")
                
                for job_id, info in st.session_state.finetuning_status.items():
                    with st.expander(f"Job: {job_id}"):
                        col_job1, col_job2 = st.columns([3, 1])
                        
                        with col_job1:
                            st.markdown(f"**Language:** {info['language']}")
                            st.markdown(f"**Base Model:** {info['model']}")
                            st.markdown(f"**Status:** {info['status']}")
                        
                        with col_job2:
                            if st.button("Check Status", key=f"check_{job_id}"):
                                try:
                                    finetuner = OpenAIFineTuner(api_key)
                                    status = finetuner.check_status(job_id)
                                    
                                    st.json(status)
                                    
                                    # Update session state
                                    if status['model']:
                                        st.success(f"✅ Complete! Model: {status['model']}")
                                        
                                        # Register model
                                        manager.register_model(
                                            provider='openai',
                                            language=info['language'],
                                            model_id=status['model'],
                                            base_model=info['model'],
                                            task_type='all'
                                        )
                                        
                                        st.session_state.finetuning_status[job_id]['status'] = 'completed'
                                        st.session_state.finetuning_status[job_id]['model_id'] = status['model']
                                
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")

# TAB 4: My Models
with tab4:
    styled_header("My Fine-Tuned Models")
    
    models = manager.list_models()
    
    if not models:
        st.info("""
        🎓 No fine-tuned models yet.
        
        Complete the fine-tuning process to see your models here.
        """)
    
    else:
        st.success(f"✅ You have {len(models)} fine-tuned model(s)")
        
        # Filter options
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            filter_language = st.selectbox(
                "Filter by Language",
                ["All"] + get_supported_languages()
            )
        
        with col_filter2:
            filter_provider = st.selectbox(
                "Filter by Provider",
                ["All", "OpenAI", "Claude", "Gemini"]
            )
        
        # Filter models
        filtered_models = models
        if filter_language != "All":
            filtered_models = [m for m in filtered_models if m['language'] == filter_language]
        if filter_provider != "All":
            filtered_models = [m for m in filtered_models if m['provider'].lower() == filter_provider.lower()]
        
        st.divider()
        
        # Display models
        for model in filtered_models:
            with st.expander(f"🤖 {model['language']} - {model['provider']} ({model['task_type']})"):
                col_model1, col_model2 = st.columns([2, 1])
                
                with col_model1:
                    st.markdown(f"**Model ID:** `{model['model_id']}`")
                    st.markdown(f"**Base Model:** {model['base_model']}")
                    st.markdown(f"**Task Type:** {model['task_type']}")
                    st.markdown(f"**Created:** {model['created_at'][:10]}")
                    
                    if model.get('metadata'):
                        st.markdown("**Metadata:**")
                        st.json(model['metadata'])
                
                with col_model2:
                    # Copy model ID button
                    if st.button("📋 Copy ID", key=f"copy_{model['model_id']}"):
                        st.code(model['model_id'])
                        st.success("✅ Copy the model ID above!")
                    
                    # Test model button
                    if st.button("🧪 Test Model", key=f"test_{model['model_id']}"):
                        st.info("Navigate to List Words or Sentiment Bearing page to test this model")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>💡 <strong>Tip:</strong> Fine-tuned models can be used in all pages by selecting them in the model dropdown.</p>
</div>
""", unsafe_allow_html=True)