"""
Page for lexicon bootstrapping using semantic similarity.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from gui.styles.custom_css import apply_custom_css, styled_header
from models.llm_factory import LLMFactory
from tasks.lexicon_bootstrapping import LexiconBootstrapping
from config.settings import (
    MODELS_CONFIG,
    BOOTSTRAP_MAX_ITERATIONS,
    BOOTSTRAP_SIMILARITY_THRESHOLD
)
from config.languages import get_supported_languages, get_seed_words

st.set_page_config(
    page_title="Lexicon Bootstrapping",
    page_icon="🔄",
    layout="wide"
)

apply_custom_css()

# Initialize session state
if 'bootstrap_lexicon' not in st.session_state:
    st.session_state.bootstrap_lexicon = []
if 'bootstrap_stats' not in st.session_state:
    st.session_state.bootstrap_stats = {}

# Header
st.title("🔄 Lexicon Bootstrapping")
st.markdown("""
Expand lexicons using **semantic similarity** and **iterative growth**.
Start with high-confidence seed words and automatically discover similar words.
""")

st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model settings
    st.subheader("Model Settings")
    provider = st.selectbox(
        "LLM Provider",
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
        key="bootstrap_api_key"
    )
    
    st.divider()
    
    # Bootstrapping settings
    st.subheader("Bootstrapping Settings")
    
    language = st.selectbox(
        "Language",
        get_supported_languages()
    )
    
    sentiment = st.selectbox(
        "Sentiment",
        ["positive", "negative", "neutral"]
    )
    
    with st.expander("🔧 Advanced Parameters"):
        max_iterations = st.slider(
            "Max Iterations",
            1, 20, BOOTSTRAP_MAX_ITERATIONS,
            help="Maximum bootstrapping iterations"
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            0.5, 0.95, BOOTSTRAP_SIMILARITY_THRESHOLD, 0.05,
            help="Minimum similarity to add word"
        )
        
        min_confidence = st.slider(
            "Min Confidence",
            0.0, 1.0, 0.6, 0.1,
            help="Minimum confidence for candidates"
        )
        
        expansion_rate = st.slider(
            "Expansion Rate",
            5, 50, 10,
            help="Words to add per iteration"
        )

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🌱 Seed Words", "📤 Candidates", "🚀 Bootstrap", "📊 Results"])

with tab1:
    styled_header("Seed Words")
    
    st.markdown("""
    Seed words are high-confidence words that form the foundation of the lexicon.
    The algorithm will find words semantically similar to these seeds.
    """)
    
    col_seed1, col_seed2 = st.columns([2, 1])
    
    with col_seed1:
        seed_method = st.radio(
            "Seed Source",
            ["Use Default Seeds", "Custom Seeds", "From Labeled Words"],
            horizontal=True
        )
    
    seed_words_list = []
    
    if seed_method == "Use Default Seeds":
        default_seeds = get_seed_words(language.lower(), sentiment)
        seed_words_list = default_seeds
        
        st.success(f"✅ Loaded {len(seed_words_list)} default seed words")
        
        st.markdown("**Default Seed Words:**")
        st.write(", ".join(seed_words_list))
    
    elif seed_method == "Custom Seeds":
        custom_seeds = st.text_area(
            "Enter seed words (one per line)",
            height=200,
            placeholder="thabo\nkatlego\nkhotso"
        )
        
        if custom_seeds:
            seed_words_list = [s.strip() for s in custom_seeds.split('\n') if s.strip()]
            st.info(f"📝 {len(seed_words_list)} seed words entered")
    
    else:  # From labeled words
        if st.session_state.get('labeled_words'):
            min_seed_confidence = st.slider(
                "Minimum Confidence for Seeds",
                0.7, 1.0, 0.85, 0.05
            )
            
            labeled = st.session_state.labeled_words
            high_conf_words = [
                w['word'] for w in labeled
                if w.get('sentiment') == sentiment
                and w.get('confidence_score', 0) >= min_seed_confidence
            ]
            
            seed_words_list = high_conf_words[:20]  # Top 20
            st.success(f"✅ Selected {len(seed_words_list)} high-confidence seeds")
        else:
            st.warning("⚠️ No labeled words found. Please label words first or use default seeds.")
    
    # Store in session state
    if 'seed_words' not in st.session_state:
        st.session_state.seed_words = []
    st.session_state.seed_words = seed_words_list

with tab2:
    styled_header("Candidate Words")
    
    st.markdown("""
    Provide a pool of candidate words that the algorithm will evaluate for similarity to seed words.
    """)
    
    candidate_method = st.radio(
        "Candidate Source",
        ["Use Generated Words", "Upload File"],
        horizontal=True
    )
    
    candidate_words = []
    
    if candidate_method == "Use Generated Words":
        if st.session_state.get('generated_words'):
            candidate_words = st.session_state.generated_words
            st.success(f"✅ Loaded {len(candidate_words)} candidate words")
        else:
            st.warning("⚠️ No generated words found. Generate words first or upload a file.")
    
    else:
        uploaded = st.file_uploader(
            "Upload candidate words",
            type=['csv', 'json']
        )
        
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                    candidate_words = df.to_dict('records')
                else:
                    import json
                    candidate_words = json.loads(uploaded.read())
                
                st.success(f"✅ Loaded {len(candidate_words)} candidates")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Preview
    if candidate_words:
        st.divider()
        preview_df = pd.DataFrame(candidate_words[:20])
        st.dataframe(preview_df, use_container_width=True)
    
    st.session_state.candidate_words = candidate_words

with tab3:
    styled_header("Run Bootstrapping")
    
    # Check prerequisites
    can_bootstrap = (
        st.session_state.get('seed_words') and
        len(st.session_state.seed_words) >= 5 and
        st.session_state.get('candidate_words') and
        api_key
    )
    
    if not can_bootstrap:
        st.warning("⚠️ Prerequisites not met:")
        if not st.session_state.get('seed_words') or len(st.session_state.seed_words) < 5:
            st.error("❌ Need at least 5 seed words")
        if not st.session_state.get('candidate_words'):
            st.error("❌ Need candidate words")
        if not api_key:
            st.error("❌ Need API key")
    else:
        st.success("✅ Ready to bootstrap!")
        
        col_boot1, col_boot2 = st.columns([2, 1])
        
        with col_boot1:
            st.info(f"""
            **Configuration:**
            - Seed words: {len(st.session_state.seed_words)}
            - Candidates: {len(st.session_state.candidate_words)}
            - Max iterations: {max_iterations}
            - Similarity threshold: {similarity_threshold}
            """)
        
        with col_boot2:
            if st.button("🚀 Start Bootstrapping", type="primary", use_container_width=True):
                try:
                    # Initialize client
                    client = LLMFactory.create_client(
                        provider=provider_key,
                        api_key=api_key,
                        model=model
                    )
                    
                    # Initialize bootstrapping
                    with st.spinner("Initializing bootstrapping..."):
                        bootstrap = LexiconBootstrapping(
                            llm_client=client,
                            language=language,
                            sentiment=sentiment,
                            seed_words=st.session_state.seed_words
                        )
                    
                    # Run bootstrapping
                    st.info("🌱 Running bootstrapping algorithm...")
                    
                    lexicon, stats = bootstrap.bootstrap(
                        candidate_words=st.session_state.candidate_words,
                        max_iterations=max_iterations,
                        similarity_threshold=similarity_threshold,
                        min_confidence=min_confidence,
                        expansion_rate=expansion_rate
                    )
                    
                    # Store results
                    st.session_state.bootstrap_lexicon = lexicon
                    st.session_state.bootstrap_stats = stats
                    st.session_state.bootstrap_growth = bootstrap.visualize_growth()
                    
                    st.success(f"✅ Bootstrapping complete! Lexicon expanded to {len(lexicon)} words")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

with tab4:
    styled_header("Bootstrapping Results")
    
    if not st.session_state.get('bootstrap_lexicon'):
        st.info("No results yet. Run bootstrapping first.")
    else:
        lexicon = st.session_state.bootstrap_lexicon
        stats = st.session_state.bootstrap_stats
        
        # Statistics
        st.markdown("#### 📈 Statistics")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Total Words", stats.get('total_words', 0))
        
        with col_stat2:
            st.metric("Seed Words", stats.get('seed_words', 0))
        
        with col_stat3:
            st.metric("Bootstrapped", stats.get('bootstrapped_words', 0))
        
        with col_stat4:
            st.metric("Expansion Ratio", f"{stats.get('expansion_ratio', 0)}x")
        
        # Growth visualization
        if st.session_state.get('bootstrap_growth'):
            st.divider()
            st.markdown("#### 📊 Lexicon Growth")
            
            growth = st.session_state.bootstrap_growth
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=growth['iterations'],
                y=growth['lexicon_sizes'],
                mode='lines+markers',
                name='Lexicon Size',
                line=dict(color='#BB86FC', width=3)
            ))
            
            fig.update_layout(
                title="Lexicon Size Over Iterations",
                xaxis_title="Iteration",
                yaxis_title="Number of Words",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Iteration history
        if stats.get('iteration_history'):
            st.divider()
            st.markdown("#### 🔄 Iteration History")
            
            history_df = pd.DataFrame(stats['iteration_history'])
            st.dataframe(history_df, use_container_width=True)
        
        # Lexicon table
        st.divider()
        st.markdown("#### 📋 Expanded Lexicon")
        
        # Filter
        col_lex1, col_lex2 = st.columns(2)
        
        with col_lex1:
            show_source = st.multiselect(
                "Show Source",
                ["seed", "bootstrap"],
                default=["seed", "bootstrap"]
            )
        
        with col_lex2:
            sort_lex = st.selectbox(
                "Sort By",
                ["Confidence", "Word", "Source"]
            )
        
        # Filter and sort
        filtered_lex = [
            w for w in lexicon
            if w.get('source') in show_source
        ]
        
        if sort_lex == "Confidence":
            filtered_lex.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
        elif sort_lex == "Word":
            filtered_lex.sort(key=lambda x: x.get('word', ''))
        elif sort_lex == "Source":
            filtered_lex.sort(key=lambda x: (x.get('source', ''), x.get('word', '')))
        
        # Display
        lex_df = pd.DataFrame(filtered_lex)
        st.dataframe(lex_df, use_container_width=True, height=400)
        
        # Export
        st.divider()
        st.markdown("#### 📥 Export Lexicon")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            csv_data = lex_df.to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                csv_data,
                f"bootstrap_lexicon_{language}_{sentiment}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col_exp2:
            import json
            json_data = json.dumps({
                'lexicon': filtered_lex,
                'statistics': stats
            }, indent=2, ensure_ascii=False)
            
            st.download_button(
                "📄 Download JSON",
                json_data,
                f"bootstrap_lexicon_{language}_{sentiment}.json",
                "application/json",
                use_container_width=True
            )
        
        with col_exp3:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                lex_df.to_excel(writer, sheet_name='Lexicon', index=False)
                pd.DataFrame([stats]).to_excel(writer, sheet_name='Statistics', index=False)
                if stats.get('iteration_history'):
                    pd.DataFrame(stats['iteration_history']).to_excel(
                        writer, sheet_name='History', index=False
                    )
            
            st.download_button(
                "📄 Download Excel",
                output.getvalue(),
                f"bootstrap_lexicon_{language}_{sentiment}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>💡 <strong>Tip:</strong> Higher similarity threshold = stricter expansion. Start with 0.7 and adjust.</p>
</div>
""", unsafe_allow_html=True)