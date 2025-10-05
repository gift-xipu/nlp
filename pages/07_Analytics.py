"""
Analytics and comparison dashboard for lexicons.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent.parent))

from gui.styles.custom_css import apply_custom_css, styled_header
from config.languages import get_supported_languages

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

apply_custom_css()

# Header
st.title("📈 Analytics Dashboard")
st.markdown("""
Compare lexicons, analyze sentiment distributions, and track generation metrics.
""")

st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Analytics Options")
    
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Lexicon Comparison", "Sentiment Distribution", "Quality Metrics", "Word Statistics"]
    )

# Main content
if analysis_type == "Lexicon Comparison":
    styled_header("Lexicon Comparison")
    
    st.markdown("Compare multiple lexicons side-by-side")
    
    # Load lexicons to compare
    col_comp1, col_comp2 = st.columns(2)
    
    lexicons_to_compare = []
    lexicon_names = []
    
    with col_comp1:
        st.markdown("#### Lexicon 1")
        
        if st.button("Load Generated Words", key="comp1_gen"):
            if st.session_state.get('generated_words'):
                lexicons_to_compare.append(st.session_state.generated_words)
                lexicon_names.append("Generated Words")
                st.success("✅ Loaded")
        
        if st.button("Load Labeled Words", key="comp1_lab"):
            if st.session_state.get('labeled_words'):
                lexicons_to_compare.append(st.session_state.labeled_words)
                lexicon_names.append("Labeled Words")
                st.success("✅ Loaded")
        
        upload1 = st.file_uploader("Upload Lexicon 1", type=['csv', 'json'], key="up1")
        if upload1:
            try:
                if upload1.name.endswith('.csv'):
                    df = pd.read_csv(upload1)
                else:
                    import json
                    df = pd.DataFrame(json.loads(upload1.read()))
                
                lexicons_to_compare.append(df.to_dict('records'))
                lexicon_names.append(upload1.name)
                st.success("✅ Loaded")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col_comp2:
        st.markdown("#### Lexicon 2")
        
        if st.button("Load Bootstrap Lexicon", key="comp2_boot"):
            if st.session_state.get('bootstrap_lexicon'):
                lexicons_to_compare.append(st.session_state.bootstrap_lexicon)
                lexicon_names.append("Bootstrap Lexicon")
                st.success("✅ Loaded")
        
        upload2 = st.file_uploader("Upload Lexicon 2", type=['csv', 'json'], key="up2")
        if upload2:
            try:
                if upload2.name.endswith('.csv'):
                    df = pd.read_csv(upload2)
                else:
                    import json
                    df = pd.DataFrame(json.loads(upload2.read()))
                
                lexicons_to_compare.append(df.to_dict('records'))
                lexicon_names.append(upload2.name)
                st.success("✅ Loaded")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Comparison
    if len(lexicons_to_compare) >= 2:
        st.divider()
        st.markdown("### Comparison Results")
        
        # Size comparison
        col_size1, col_size2, col_size3 = st.columns(3)
        
        with col_size1:
            st.metric("Lexicon 1 Size", len(lexicons_to_compare[0]))
        
        with col_size2:
            st.metric("Lexicon 2 Size", len(lexicons_to_compare[1]))
        
        with col_size3:
            # Overlap
            words1 = set([w.get('word', '') for w in lexicons_to_compare[0]])
            words2 = set([w.get('word', '') for w in lexicons_to_compare[1]])
            overlap = len(words1 & words2)
            st.metric("Overlap", overlap)
        
        # Venn diagram
        import plotly.graph_objects as go
        
        unique1 = len(words1 - words2)
        unique2 = len(words2 - words1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name=lexicon_names[0],
            x=['Unique to Lex1', 'Shared', 'Unique to Lex2'],
            y=[unique1, overlap, unique2],
            marker_color='#BB86FC'
        ))
        
        fig.update_layout(
            title="Word Distribution",
            yaxis_title="Count",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment comparison
        if all('sentiment' in w for lex in lexicons_to_compare for w in lex[:5]):
            st.divider()
            st.markdown("### Sentiment Comparison")
            
            sentiments1 = Counter([w.get('sentiment') for w in lexicons_to_compare[0]])
            sentiments2 = Counter([w.get('sentiment') for w in lexicons_to_compare[1]])
            
            comparison_df = pd.DataFrame({
                lexicon_names[0]: [sentiments1.get('positive', 0), sentiments1.get('negative', 0), sentiments1.get('neutral', 0)],
                lexicon_names[1]: [sentiments2.get('positive', 0), sentiments2.get('negative', 0), sentiments2.get('neutral', 0)]
            }, index=['Positive', 'Negative', 'Neutral'])
            
            fig_sent = go.Figure()
            
            fig_sent.add_trace(go.Bar(
                name=lexicon_names[0],
                x=comparison_df.index,
                y=comparison_df[lexicon_names[0]],
                marker_color='#BB86FC'
            ))
            
            fig_sent.add_trace(go.Bar(
                name=lexicon_names[1],
                x=comparison_df.index,
                y=comparison_df[lexicon_names[1]],
                marker_color='#90CAF9'
            ))
            
            fig_sent.update_layout(
                title="Sentiment Distribution Comparison",
                xaxis_title="Sentiment",
                yaxis_title="Count",
                barmode='group',
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig_sent, use_container_width=True)

elif analysis_type == "Sentiment Distribution":
    styled_header("Sentiment Distribution Analysis")
    
    # Select data source
    data_source = st.selectbox(
        "Data Source",
        ["Labeled Words", "Bootstrap Lexicon", "Upload File"]
    )
    
    lexicon_data = []
    
    if data_source == "Labeled Words" and st.session_state.get('labeled_words'):
        lexicon_data = st.session_state.labeled_words
    elif data_source == "Bootstrap Lexicon" and st.session_state.get('bootstrap_lexicon'):
        lexicon_data = st.session_state.bootstrap_lexicon
    elif data_source == "Upload File":
        uploaded = st.file_uploader("Upload lexicon", type=['csv', 'json'])
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                    lexicon_data = df.to_dict('records')
                else:
                    import json
                    lexicon_data = json.loads(uploaded.read())
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    if lexicon_data:
        sentiments = [w.get('sentiment', 'unknown') for w in lexicon_data]
        sentiment_counts = Counter(sentiments)
        
        # Pie chart
        col_pie1, col_pie2 = st.columns([2, 1])
        
        with col_pie1:
            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=list(sentiment_counts.keys()),
                    values=list(sentiment_counts.values()),
                    marker_colors=['#A5D6B7', '#FF5370', '#90CAF9']
                )
            ])
            
            fig_pie.update_layout(
                title="Sentiment Distribution",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_pie2:
            st.markdown("### Counts")
            for sentiment, count in sentiment_counts.most_common():
                pct = count / len(sentiments) * 100
                st.metric(sentiment.title(), f"{count} ({pct:.1f}%)")
        
        # Confidence distribution by sentiment
        if all('confidence_score' in w for w in lexicon_data[:10]):
            st.divider()
            st.markdown("### Confidence Distribution by Sentiment")
            
            df_conf = pd.DataFrame(lexicon_data)
            
            fig_box = px.box(
                df_conf,
                x='sentiment',
                y='confidence_score',
                color='sentiment',
                color_discrete_map={
                    'positive': '#A5D6B7',
                    'negative': '#FF5370',
                    'neutral': '#90CAF9'
                }
            )
            
            fig_box.update_layout(
                title="Confidence Scores by Sentiment",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig_box, use_container_width=True)

elif analysis_type == "Quality Metrics":
    styled_header("Quality Metrics Analysis")
    
    if st.session_state.get('validation_report'):
        reports = st.session_state.validation_report
        quality = reports.get('quality', {})
        
        # Radar chart
        categories = ['Validity', 'Uniqueness', 'Balance', 'Confidence']
        scores = [
            quality.get('validity_score', 0),
            quality.get('uniqueness_score', 0),
            quality.get('balance_score', 0),
            quality.get('confidence_score', 0)
        ]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            line_color='#BB86FC',
            fillcolor='rgba(187, 134, 252, 0.3)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            template="plotly_dark",
            title="Quality Metrics Radar",
            height=500
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Metrics table
        st.divider()
        
        metrics_df = pd.DataFrame({
            'Metric': categories,
            'Score': scores,
            'Status': ['✅' if s >= 80 else '⚠️' if s >= 60 else '❌' for s in scores]
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    else:
        st.info("Run validation first to see quality metrics")

else:  # Word Statistics
    styled_header("Word Statistics")
    
    # Select data source
    data_source = st.selectbox(
        "Data Source",
        ["Generated Words", "Labeled Words", "Bootstrap Lexicon"],
        key="word_stats_source"
    )
    
    lexicon_data = []
    
    if data_source == "Generated Words":
        lexicon_data = st.session_state.get('generated_words', [])
    elif data_source == "Labeled Words":
        lexicon_data = st.session_state.get('labeled_words', [])
    else:
        lexicon_data = st.session_state.get('bootstrap_lexicon', [])
    
    if lexicon_data:
        # Word length distribution
        word_lengths = [len(w.get('word', '')) for w in lexicon_data]
        
        fig_hist = go.Figure(data=[
            go.Histogram(x=word_lengths, nbinsx=20, marker_color='#BB86FC')
        ])
        
        fig_hist.update_layout(
            title="Word Length Distribution",
            xaxis_title="Word Length (characters)",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Statistics
        st.divider()
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        import numpy as np
        
        with col_stat1:
            st.metric("Total Words", len(lexicon_data))
        
        with col_stat2:
            st.metric("Avg Length", f"{np.mean(word_lengths):.1f}")
        
        with col_stat3:
            st.metric("Min Length", min(word_lengths))
        
        with col_stat4:
            st.metric("Max Length", max(word_lengths))
        
        # Character frequency
        st.divider()
        st.markdown("### Character Frequency")
        
        all_chars = ''.join([w.get('word', '') for w in lexicon_data])
        char_freq = Counter(all_chars.lower())
        
        # Top 20 characters
        top_chars = char_freq.most_common(20)
        
        fig_chars = go.Figure(data=[
            go.Bar(
                x=[c[0] for c in top_chars],
                y=[c[1] for c in top_chars],
                marker_color='#BB86FC'
            )
        ])
        
        fig_chars.update_layout(
            title="Top 20 Characters",
            xaxis_title="Character",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig_chars, use_container_width=True)
    
    else:
        st.info("No lexicon data available")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>💡 <strong>Tip:</strong> Compare different prompt strategies to find the best approach.</p>
</div>
""", unsafe_allow_html=True)