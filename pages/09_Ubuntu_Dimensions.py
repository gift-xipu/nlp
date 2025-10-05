"""
Ubuntu Sentiment Dimensions - Visualization & Analysis

Revolutionary feature: Multi-dimensional sentiment analysis based on Ubuntu philosophy.
Goes beyond positive/negative to capture communal, relational aspects of emotions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from gui.styles.custom_css import apply_custom_css, styled_header
from gui.components.model_selector import render_model_selector
from models.llm_factory import LLMFactory
from tasks.ubuntu_dimensions import UbuntuSentimentAnalyzer, UbuntuSentiment, calculate_ubuntu_score
from config.languages import get_supported_languages

st.set_page_config(
    page_title="Ubuntu Dimensions",
    page_icon="🌍",
    layout="wide"
)

apply_custom_css()

# Initialize session state
if 'ubuntu_results' not in st.session_state:
    st.session_state.ubuntu_results = []

# Header
st.title("🌍 Ubuntu Sentiment Dimensions")
st.markdown("""
**Revolutionary Multi-Dimensional Sentiment Analysis**

Beyond positive/negative: Analyze emotions through the lens of Ubuntu philosophy.
Capture communal, relational, and temporal aspects of sentiment in African languages.

*Research Innovation: First computational framework for collectivist sentiment dimensions*
""")

st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    language = st.selectbox(
        "Language",
        get_supported_languages()
    )
    
    st.divider()
    
    # Model selection
    provider_key, model, api_key, use_finetuned, finetuned_model_id = render_model_selector(
        provider_options=["OpenAI", "Claude", "Gemini"],
        language=language,
        task_type='sentiment_bearing',
        key_prefix="ubuntu"
    )
    
    st.divider()
    
    # Info
    with st.expander("📖 About Ubuntu Dimensions"):
        st.markdown("""
        **Traditional NLP:** positive/negative only
        
        **Ubuntu NLP:** 6 dimensions capturing:
        - Individual ↔ Communal
        - Active ↔ Passive  
        - Immediate ↔ Ancestral
        - Harmonizing ↔ Dividing
        - Internal ↔ Relational
        - Plus traditional valence
        
        *"I am because we are"* - Ubuntu philosophy
        """)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Analyze Words",
    "📊 Visualize Dimensions",
    "🎯 Compare Cultures",
    "📈 Research Data"
])

# TAB 1: Analyze Words
with tab1:
    styled_header("Analyze Words with Ubuntu Dimensions")
    
    col_input1, col_input2 = st.columns([2, 1])
    
    with col_input1:
        input_method = st.radio(
            "Input Method",
            ["Single Word", "Batch Upload", "Use Generated Words"],
            horizontal=True
        )
    
    words_to_analyze = []
    
    if input_method == "Single Word":
        col_word1, col_word2 = st.columns(2)
        
        with col_word1:
            word = st.text_input("Word", placeholder="thabo")
        
        with col_word2:
            translation = st.text_input("Translation", placeholder="joy, happiness")
        
        if word and translation:
            words_to_analyze = [{'word': word, 'translation': translation}]
    
    elif input_method == "Batch Upload":
        uploaded = st.file_uploader("Upload CSV/JSON", type=['csv', 'json'])
        
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                    words_to_analyze = df.to_dict('records')
                else:
                    import json
                    words_to_analyze = json.loads(uploaded.read())
                
                st.success(f"✅ Loaded {len(words_to_analyze)} words")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    else:  # Use Generated Words
        if st.session_state.get('generated_words'):
            words_to_analyze = st.session_state.generated_words[:20]
            st.info(f"📝 Using {len(words_to_analyze)} generated words")
        else:
            st.warning("⚠️ No generated words found. Generate words first.")
    
    if words_to_analyze:
        st.divider()
        
        if st.button("🌍 Analyze with Ubuntu Dimensions", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ Please enter API key")
            else:
                try:
                    # Create client
                    client = LLMFactory.create_client(
                        provider=provider_key,
                        api_key=api_key,
                        model=model if not use_finetuned else None,
                        use_finetuned=use_finetuned,
                        finetuned_model_id=finetuned_model_id
                    )
                    
                    # Create analyzer
                    analyzer = UbuntuSentimentAnalyzer(client)
                    
                    # Progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(current, total):
                        progress_bar.progress(current / total)
                        status_text.info(f"🌍 Analyzing {current}/{total} words...")
                    
                    # Analyze
                    results = analyzer.batch_analyze(
                        words_to_analyze,
                        language,
                        update_progress
                    )
                    
                    # Store results
                    st.session_state.ubuntu_results = results
                    
                    progress_bar.progress(1.0)
                    status_text.success(f"✅ Analyzed {len(results)} words!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

# TAB 2: Visualize
with tab2:
    styled_header("Multi-Dimensional Visualization")
    
    if not st.session_state.ubuntu_results:
        st.info("👈 Analyze words first in the 'Analyze Words' tab")
    else:
        results = st.session_state.ubuntu_results
        
        # Select word to visualize
        word_options = [f"{r.word} ({r.translation})" for r in results]
        selected_word = st.selectbox("Select Word to Visualize", word_options)
        
        idx = word_options.index(selected_word)
        sentiment = results[idx]
        
        st.divider()
        
        # Two columns: Radar + Profile
        col_viz1, col_viz2 = st.columns([1, 1])
        
        with col_viz1:
            st.markdown("### 🎯 Dimension Radar")
            
            # Radar chart
            dimensions = [
                'Valence',
                'Communal',
                'Active',
                'Ancestral',
                'Harmonizing',
                'Relational'
            ]
            
            values = [
                sentiment.valence,
                sentiment.individual_communal,
                sentiment.active_passive,
                sentiment.immediate_ancestral,
                sentiment.harmonizing_dividing,
                sentiment.internal_relational
            ]
            
            # Normalize to 0-1 for visualization
            normalized_values = [(v + 1) / 2 for v in values]
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=dimensions,
                fill='toself',
                name=sentiment.word,
                line=dict(color='#BB86FC', width=3),
                fillcolor='rgba(187, 134, 252, 0.3)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=False,
                template="plotly_dark",
                height=400,
                title=f"{sentiment.word} - Ubuntu Profile"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col_viz2:
            st.markdown("### 📋 Ubuntu Profile")
            
            # Ubuntu score
            ubuntu_score = calculate_ubuntu_score(sentiment)
            
            st.metric(
                "Ubuntu Score",
                f"{ubuntu_score:.2f}",
                help="How aligned with Ubuntu values (-1 to +1)"
            )
            
            # Profile
            profile = sentiment.get_ubuntu_profile()
            
            for key, value in profile.items():
                st.markdown(f"**{key.title()}:**")
                st.info(value)
        
        # Details below
        st.divider()
        
        col_detail1, col_detail2 = st.columns(2)
        
        with col_detail1:
            st.markdown("### 💭 Analysis")
            st.write(f"**Reasoning:** {sentiment.reasoning}")
            
            if sentiment.cultural_notes:
                st.write(f"**Cultural Context:** {sentiment.cultural_notes}")
        
        with col_detail2:
            st.markdown("### 📝 Example Usage")
            if sentiment.example_usage:
                st.code(sentiment.example_usage, language=None)
            else:
                st.info("No example provided")
        
        # Raw scores
        st.divider()
        
        with st.expander("🔢 Raw Dimension Scores"):
            scores_df = pd.DataFrame({
                'Dimension': dimensions,
                'Score': values,
                'Range': ['(-1 to +1)'] * len(dimensions)
            })
            st.dataframe(scores_df, use_container_width=True, hide_index=True)

# TAB 3: Compare
with tab3:
    styled_header("Cross-Cultural Comparison")
    
    if len(st.session_state.ubuntu_results) < 2:
        st.info("👈 Analyze at least 2 words to compare")
    else:
        results = st.session_state.ubuntu_results
        
        col_comp1, col_comp2 = st.columns(2)
        
        word_options = [f"{r.word} ({r.translation})" for r in results]
        
        with col_comp1:
            word1_select = st.selectbox("Word 1", word_options, key="comp_word1")
        
        with col_comp2:
            word2_select = st.selectbox("Word 2", word_options, key="comp_word2", index=min(1, len(word_options)-1))
        
        if word1_select != word2_select:
            idx1 = word_options.index(word1_select)
            idx2 = word_options.index(word2_select)
            
            sent1 = results[idx1]
            sent2 = results[idx2]
            
            st.divider()
            
            # Comparison visualization
            st.markdown("### 📊 Dimension Comparison")
            
            dimensions = [
                'Valence', 'Communal', 'Active',
                'Ancestral', 'Harmonizing', 'Relational'
            ]
            
            values1 = [
                sent1.valence, sent1.individual_communal,
                sent1.active_passive, sent1.immediate_ancestral,
                sent1.harmonizing_dividing, sent1.internal_relational
            ]
            
            values2 = [
                sent2.valence, sent2.individual_communal,
                sent2.active_passive, sent2.immediate_ancestral,
                sent2.harmonizing_dividing, sent2.internal_relational
            ]
            
            # Normalize
            norm1 = [(v + 1) / 2 for v in values1]
            norm2 = [(v + 1) / 2 for v in values2]
            
            fig_comp = go.Figure()
            
            fig_comp.add_trace(go.Scatterpolar(
                r=norm1,
                theta=dimensions,
                fill='toself',
                name=sent1.word,
                line=dict(color='#BB86FC', width=2),
                fillcolor='rgba(187, 134, 252, 0.2)'
            ))
            
            fig_comp.add_trace(go.Scatterpolar(
                r=norm2,
                theta=dimensions,
                fill='toself',
                name=sent2.word,
                line=dict(color='#90CAF9', width=2),
                fillcolor='rgba(144, 202, 249, 0.2)'
            ))
            
            fig_comp.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template="plotly_dark",
                height=500,
                title="Cultural Sentiment Profiles"
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Similarity metrics
            st.divider()
            st.markdown("### 📏 Similarity Metrics")
            
            from tasks.ubuntu_dimensions import UbuntuSentimentAnalyzer
            analyzer = UbuntuSentimentAnalyzer(None)
            comparison = analyzer.compare_cultural_profiles(sent1, sent2)
            
            col_met1, col_met2, col_met3 = st.columns(3)
            
            with col_met1:
                st.metric("Overall Similarity", f"{2 - comparison['overall_similarity']:.2f}")
            
            with col_met2:
                st.metric("Communal Difference", f"{comparison['communal_diff']:.2f}")
            
            with col_met3:
                st.metric("Relational Difference", f"{comparison['relational_diff']:.2f}")

# TAB 4: Research Data
with tab4:
    styled_header("Research Data Export")
    
    if not st.session_state.ubuntu_results:
        st.info("👈 Analyze words first")
    else:
        results = st.session_state.ubuntu_results
        
        st.success(f"✅ {len(results)} words analyzed with Ubuntu dimensions")
        
        # Convert to DataFrame
        data = []
        for r in results:
            data.append({
                'word': r.word,
                'translation': r.translation,
                'language': r.language,
                'valence': r.valence,
                'valence_confidence': r.valence_confidence,
                'individual_communal': r.individual_communal,
                'active_passive': r.active_passive,
                'immediate_ancestral': r.immediate_ancestral,
                'harmonizing_dividing': r.harmonizing_dividing,
                'internal_relational': r.internal_relational,
                'ubuntu_score': calculate_ubuntu_score(r),
                'reasoning': r.reasoning,
                'cultural_notes': r.cultural_notes or '',
                'example_usage': r.example_usage or ''
            })
        
        df = pd.DataFrame(data)
        
        # Show table
        st.dataframe(df, use_container_width=True, height=400)
        
        st.divider()
        
        # Statistics
        st.markdown("### 📊 Dataset Statistics")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            avg_ubuntu = df['ubuntu_score'].mean()
            st.metric("Avg Ubuntu Score", f"{avg_ubuntu:.2f}")
        
        with col_stat2:
            avg_communal = df['individual_communal'].mean()
            st.metric("Avg Communal", f"{avg_communal:.2f}")
        
        with col_stat3:
            avg_harmony = df['harmonizing_dividing'].mean()
            st.metric("Avg Harmonizing", f"{avg_harmony:.2f}")
        
        with col_stat4:
            avg_relational = df['internal_relational'].mean()
            st.metric("Avg Relational", f"{avg_relational:.2f}")
        
        st.divider()
        
        # Distribution plots
        st.markdown("### 📈 Dimension Distributions")
        
        dimension_cols = [
            'individual_communal', 'active_passive',
            'immediate_ancestral', 'harmonizing_dividing',
            'internal_relational'
        ]
        
        dimension_names = [
            'Individual ↔ Communal', 'Passive ↔ Active',
            'Immediate ↔ Ancestral', 'Dividing ↔ Harmonizing',
            'Internal ↔ Relational'
        ]
        
        fig_dist = go.Figure()
        
        for col, name in zip(dimension_cols, dimension_names):
            fig_dist.add_trace(go.Box(
                y=df[col],
                name=name,
                boxmean='sd'
            ))
        
        fig_dist.update_layout(
            title="Ubuntu Dimension Distributions",
            yaxis_title="Score (-1 to +1)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.divider()
        
        # Export options
        st.markdown("### 📥 Export Research Data")
        
        col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
        
        with col_exp1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📄 CSV",
                csv_data,
                f"ubuntu_dimensions_{language}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col_exp2:
            import json
            json_data = json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False)
            st.download_button(
                "📄 JSON",
                json_data,
                f"ubuntu_dimensions_{language}.json",
                "application/json",
                use_container_width=True
            )
        
        with col_exp3:
            # Excel with multiple sheets
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Ubuntu Dimensions', index=False)
                
                # Statistics sheet
                stats_df = pd.DataFrame({
                    'Metric': ['Total Words', 'Language', 'Avg Ubuntu Score', 
                              'Avg Communal', 'Avg Harmonizing', 'Avg Relational'],
                    'Value': [len(results), language, f"{avg_ubuntu:.2f}",
                             f"{avg_communal:.2f}", f"{avg_harmony:.2f}", 
                             f"{avg_relational:.2f}"]
                })
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            st.download_button(
                "📄 Excel",
                output.getvalue(),
                f"ubuntu_dimensions_{language}.xlsx",
                use_container_width=True
            )
        
        with col_exp4:
            # Research-ready format (for publications)
            research_text = f"""# Ubuntu Sentiment Dimensions Dataset

Language: {language}
Total Words: {len(results)}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}

## Summary Statistics
- Average Ubuntu Score: {avg_ubuntu:.2f}
- Average Communal Dimension: {avg_communal:.2f}
- Average Harmonizing Dimension: {avg_harmony:.2f}
- Average Relational Dimension: {avg_relational:.2f}

## Dimensions
1. Valence: Traditional positive/negative (-1 to +1)
2. Individual-Communal: Personal vs collective emotion (-1 to +1)
3. Active-Passive: Requires action vs passively experienced (-1 to +1)
4. Immediate-Ancestral: Present vs eternal/ancestral (-1 to +1)
5. Harmonizing-Dividing: Unifies vs divides community (-1 to +1)
6. Internal-Relational: Personal feeling vs social bond (-1 to +1)

## Data
"""
            for r in results:
                research_text += f"\n{r.word} ({r.translation})\n"
                research_text += f"  Ubuntu Score: {calculate_ubuntu_score(r):.2f}\n"
                research_text += f"  Dimensions: [{r.valence:.2f}, {r.individual_communal:.2f}, "
                research_text += f"{r.active_passive:.2f}, {r.immediate_ancestral:.2f}, "
                research_text += f"{r.harmonizing_dividing:.2f}, {r.internal_relational:.2f}]\n"
                research_text += f"  Reasoning: {r.reasoning}\n"
                if r.cultural_notes:
                    research_text += f"  Cultural Notes: {r.cultural_notes}\n"
            
            st.download_button(
                "📄 TXT (Research)",
                research_text,
                f"ubuntu_research_{language}.txt",
                "text/plain",
                use_container_width=True
            )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p><strong>🌍 Ubuntu Sentiment Dimensions</strong></p>
    <p>Revolutionary multi-dimensional sentiment analysis for African languages</p>
    <p><em>"I am because we are" - Ubuntu Philosophy → Computational Linguistics</em></p>
    <p style='margin-top: 15px; font-size: 0.85em;'>
        Research Innovation: First framework for collectivist sentiment dimensions<br>
        Beyond Western positive/negative → Capturing communal, relational, ancestral aspects
    </p>
</div>
""", unsafe_allow_html=True)