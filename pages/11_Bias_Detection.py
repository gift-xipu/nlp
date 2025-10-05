"""
Bias Detection Lab - Audit Sentiment Analysis Systems

Compare your lexicons against existing systems (GPT, Google Translate, etc.)
Quantify bias, identify gaps, and document "semantic colonization"

Research Innovation: First systematic bias audit for African language sentiment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import spearmanr

from gui.styles.custom_css import apply_custom_css, styled_header
from gui.components.model_selector import render_model_selector
from models.llm_factory import LLMFactory
from config.languages import get_supported_languages

st.set_page_config(
    page_title="Bias Detection Lab",
    page_icon="🔬",
    layout="wide"
)

apply_custom_css()

# Initialize session state
if 'bias_results' not in st.session_state:
    st.session_state.bias_results = None

# Header
st.title("🔬 Bias Detection Lab")
st.markdown("""
**Audit Sentiment Analysis Systems for African Languages**

Compare your validated lexicons against major AI systems. Identify where models fail,
quantify bias, and document "semantic colonization" - English concepts forced onto African languages.

*Research Innovation: First systematic bias audit framework*
""")

st.divider()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "⚖️ Compare Systems",
    "📊 Bias Metrics",
    "🔍 Error Analysis",
    "📝 Generate Report"
])

# TAB 1: Compare Systems
with tab1:
    styled_header("Compare Sentiment Analysis Systems")
    
    col_comp1, col_comp2 = st.columns([2, 1])
    
    with col_comp1:
        st.markdown("### 📤 Upload Your Validated Lexicon")
        
        uploaded_lexicon = st.file_uploader(
            "Upload lexicon with validated sentiments",
            type=['csv', 'json'],
            help="Must include: word, translation, validated_sentiment columns"
        )
        
        if uploaded_lexicon:
            try:
                if uploaded_lexicon.name.endswith('.csv'):
                    lexicon_df = pd.read_csv(uploaded_lexicon)
                else:
                    import json
                    lexicon_df = pd.DataFrame(json.loads(uploaded_lexicon.read()))
                
                st.success(f"✅ Loaded {len(lexicon_df)} validated words")
                
                # Show preview
                st.dataframe(lexicon_df.head(), use_container_width=True)
                
                st.session_state.lexicon_df = lexicon_df
                st.session_state.lexicon_language = lexicon_df['language'].iloc[0] if 'language' in lexicon_df else 'Unknown'
            
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with col_comp2:
        st.markdown("### ⚙️ Baseline System")
        
        baseline_system = st.selectbox(
            "Compare Against",
            ["GPT-4", "GPT-3.5", "Claude", "Gemini", "Google Translate (via API)"],
            help="Select system to audit for bias"
        )
        
        st.info("""
        **What we'll check:**
        
        ✅ Agreement rate
        ✅ Systematic biases
        ✅ Missing emotions
        ✅ Cultural misunderstandings
        """)
    
    # Run comparison
    if 'lexicon_df' in st.session_state:
        st.divider()
        
        # Model selection for baseline
        provider_key, model, api_key, use_finetuned, finetuned_model_id = render_model_selector(
            provider_options=["OpenAI", "Claude", "Gemini"],
            language=st.session_state.lexicon_language,
            task_type='sentiment_bearing',
            key_prefix="bias_detection"
        )
        
        if st.button("🔬 Run Bias Analysis", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ Enter API key")
            else:
                try:
                    lexicon_df = st.session_state.lexicon_df
                    
                    # Create client
                    client = LLMFactory.create_client(
                        provider=provider_key,
                        api_key=api_key,
                        model=model
                    )
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    baseline_predictions = []
                    total = len(lexicon_df)
                    
                    # Get baseline predictions
                    for i, row in lexicon_df.iterrows():
                        word = row['word']
                        translation = row.get('translation', '')
                        
                        status_text.info(f"🔬 Analyzing {i+1}/{total}: {word}")
                        
                        # Simple sentiment prompt
                        prompt = f"""Classify sentiment of this word as positive, negative, or neutral.

Word: {word}
Translation: {translation}

Respond with ONLY one word: positive, negative, or neutral"""
                        
                        try:
                            response = client.generate(prompt, temperature=0.1, max_tokens=10)
                            predicted = response.strip().lower()
                            
                            # Clean response
                            if 'positive' in predicted:
                                predicted = 'positive'
                            elif 'negative' in predicted:
                                predicted = 'negative'
                            else:
                                predicted = 'neutral'
                            
                            baseline_predictions.append(predicted)
                        
                        except:
                            baseline_predictions.append('unknown')
                        
                        progress_bar.progress((i + 1) / total)
                    
                    # Store results
                    results_df = lexicon_df.copy()
                    results_df['baseline_prediction'] = baseline_predictions
                    results_df['agreement'] = results_df.apply(
                        lambda row: row.get('validated_sentiment', row.get('sentiment')) == row['baseline_prediction'],
                        axis=1
                    )
                    
                    st.session_state.bias_results = {
                        'results_df': results_df,
                        'baseline_system': baseline_system,
                        'language': st.session_state.lexicon_language
                    }
                    
                    status_text.success(f"✅ Analysis complete!")
                    progress_bar.progress(1.0)
                    
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

# TAB 2: Bias Metrics
with tab2:
    styled_header("Bias Metrics & Statistics")
    
    if not st.session_state.bias_results:
        st.info("👈 Run bias analysis first in the 'Compare Systems' tab")
    else:
        results = st.session_state.bias_results
        df = results['results_df']
        baseline = results['baseline_system']
        language = results['language']
        
        # Overall metrics
        st.markdown(f"### 📊 {baseline} Performance on {language}")
        
        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        
        with col_met1:
            accuracy = df['agreement'].mean() * 100
            st.metric("Overall Accuracy", f"{accuracy:.1f}%")
        
        with col_met2:
            total = len(df)
            st.metric("Words Tested", total)
        
        with col_met3:
            correct = df['agreement'].sum()
            st.metric("Correct", correct)
        
        with col_met4:
            wrong = total - correct
            st.metric("Incorrect", wrong)
        
        st.divider()
        
        # Confusion matrix
        st.markdown("### 📊 Confusion Matrix")
        
        validated_col = 'validated_sentiment' if 'validated_sentiment' in df else 'sentiment'
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        
        sentiments = ['positive', 'negative', 'neutral']
        
        y_true = df[validated_col].values
        y_pred = df['baseline_prediction'].values
        
        cm = confusion_matrix(y_true, y_pred, labels=sentiments)
        
        # Visualize
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=sentiments,
            y=sentiments,
            colorscale='RdYlGn',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        
        fig_cm.update_layout(
            title=f"{baseline} Predictions vs Ground Truth",
            xaxis_title="Predicted Sentiment",
            yaxis_title="True Sentiment",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.divider()
        
        # Per-sentiment accuracy
        st.markdown("### 📈 Performance by Sentiment")
        
        sentiment_accuracy = []
        
        for sent in sentiments:
            mask = df[validated_col] == sent
            if mask.sum() > 0:
                acc = df[mask]['agreement'].mean() * 100
                count = mask.sum()
                sentiment_accuracy.append({'Sentiment': sent, 'Accuracy': acc, 'Count': count})
        
        acc_df = pd.DataFrame(sentiment_accuracy)
        
        col_acc1, col_acc2 = st.columns([2, 1])
        
        with col_acc1:
            fig_acc = go.Figure(data=[
                go.Bar(
                    x=acc_df['Sentiment'],
                    y=acc_df['Accuracy'],
                    text=acc_df['Accuracy'].round(1),
                    texttemplate='%{text}%',
                    textposition='outside',
                    marker_color=['#A5D6A7', '#EF5350', '#90CAF9']
                )
            ])
            
            fig_acc.update_layout(
                title="Accuracy by Sentiment Category",
                yaxis_title="Accuracy (%)",
                yaxis_range=[0, 110],
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col_acc2:
            st.dataframe(acc_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Bias patterns
        st.markdown("### 🎯 Systematic Bias Patterns")
        
        # Calculate bias direction
        pos_as_neg = len(df[(df[validated_col] == 'positive') & (df['baseline_prediction'] == 'negative')])
        neg_as_pos = len(df[(df[validated_col] == 'negative') & (df['baseline_prediction'] == 'positive')])
        
        col_bias1, col_bias2 = st.columns(2)
        
        with col_bias1:
            st.metric(
                "Positive → Negative Errors",
                pos_as_neg,
                delta=f"{pos_as_neg/len(df)*100:.1f}%",
                delta_color="inverse"
            )
            
            if pos_as_neg > 0:
                examples = df[(df[validated_col] == 'positive') & (df['baseline_prediction'] == 'negative')]['word'].head(5).tolist()
                st.caption(f"Examples: {', '.join(examples)}")
        
        with col_bias2:
            st.metric(
                "Negative → Positive Errors",
                neg_as_pos,
                delta=f"{neg_as_pos/len(df)*100:.1f}%",
                delta_color="inverse"
            )
            
            if neg_as_pos > 0:
                examples = df[(df[validated_col] == 'negative') & (df['baseline_prediction'] == 'positive')]['word'].head(5).tolist()
                st.caption(f"Examples: {', '.join(examples)}")

# TAB 3: Error Analysis
with tab3:
    styled_header("Detailed Error Analysis")
    
    if not st.session_state.bias_results:
        st.info("👈 Run bias analysis first")
    else:
        results = st.session_state.bias_results
        df = results['results_df']
        validated_col = 'validated_sentiment' if 'validated_sentiment' in df else 'sentiment'
        
        # Filter options
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            show_only = st.selectbox(
                "Show",
                ["All Words", "Errors Only", "Correct Only"],
                index=1
            )
        
        with col_filter2:
            sentiment_filter = st.selectbox(
                "Filter by True Sentiment",
                ["All"] + ['positive', 'negative', 'neutral']
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if show_only == "Errors Only":
            filtered_df = filtered_df[~filtered_df['agreement']]
        elif show_only == "Correct Only":
            filtered_df = filtered_df[filtered_df['agreement']]
        
        if sentiment_filter != "All":
            filtered_df = filtered_df[filtered_df[validated_col] == sentiment_filter]
        
        st.markdown(f"### 📋 {len(filtered_df)} Words")
        
        # Display with highlighting
        def highlight_errors(row):
            if not row['agreement']:
                return ['background-color: rgba(255, 82, 82, 0.2)'] * len(row)
            return [''] * len(row)
        
        display_df = filtered_df[[
            'word', 'translation', validated_col, 
            'baseline_prediction', 'agreement'
        ]].copy()
        
        display_df.columns = [
            'Word', 'Translation', 'True Sentiment',
            'Baseline Prediction', 'Correct'
        ]
        
        st.dataframe(
            display_df.style.apply(highlight_errors, axis=1),
            use_container_width=True,
            height=500
        )
        
        st.divider()
        
        # Cultural gap analysis
        st.markdown("### 🌍 Cultural Gap Analysis")
        
        st.markdown("""
        **Hypothesis:** Errors may indicate cultural concepts that don't translate well.
        
        Words where the baseline system fails might represent:
        - Ubuntu-specific emotions
        - Culturally-bound sentiment expressions
        - Relational sentiments missing in English
        """)
        
        errors_df = df[~df['agreement']]
        
        if len(errors_df) > 0:
            st.markdown(f"**{len(errors_df)} errors found** - Potential cultural gaps")
            
            # Show most common error patterns
            error_patterns = errors_df.groupby([validated_col, 'baseline_prediction']).size().reset_index(name='count')
            error_patterns = error_patterns.sort_values('count', ascending=False)
            
            st.dataframe(error_patterns, use_container_width=True, hide_index=True)
            
            # Export errors for annotation
            st.divider()
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                csv_errors = errors_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Errors for Review",
                    csv_errors,
                    f"bias_errors_{language}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col_export2:
                st.info("""
                **Next Steps:**
                
                1. Review errors with native speakers
                2. Document cultural context
                3. Build "Cultural Gap Corpus"
                4. Publish findings
                """)

# TAB 4: Generate Report
with tab4:
    styled_header("Generate Bias Audit Report")
    
    if not st.session_state.bias_results:
        st.info("👈 Run bias analysis first")
    else:
        results = st.session_state.bias_results
        df = results['results_df']
        baseline = results['baseline_system']
        language = results['language']
        validated_col = 'validated_sentiment' if 'validated_sentiment' in df else 'sentiment'
        
        st.markdown(f"""
        ### 📝 Bias Audit Report
        
        **System Tested:** {baseline}  
        **Language:** {language}  
        **Words Analyzed:** {len(df)}  
        **Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
        """)
        
        # Calculate all metrics
        accuracy = df['agreement'].mean() * 100
        
        sentiments = ['positive', 'negative', 'neutral']
        sentiment_accuracy = {}
        for sent in sentiments:
            mask = df[validated_col] == sent
            if mask.sum() > 0:
                sentiment_accuracy[sent] = df[mask]['agreement'].mean() * 100
        
        pos_as_neg = len(df[(df[validated_col] == 'positive') & (df['baseline_prediction'] == 'negative')])
        neg_as_pos = len(df[(df[validated_col] == 'negative') & (df['baseline_prediction'] == 'positive')])
        
        # Generate report text
        report = f"""# Bias Audit Report: {baseline} on {language}

## Executive Summary

This report documents systematic bias in {baseline}'s sentiment analysis for {language}.
Total words analyzed: {len(df)}

## Key Findings

### Overall Performance
- **Accuracy:** {accuracy:.1f}%
- **Correct Predictions:** {df['agreement'].sum()}/{len(df)}
- **Error Rate:** {100-accuracy:.1f}%

### Performance by Sentiment
"""
        
        for sent, acc in sentiment_accuracy.items():
            count = len(df[df[validated_col] == sent])
            report += f"- **{sent.title()}:** {acc:.1f}% accuracy ({count} words)\n"
        
        report += f"""

### Bias Patterns

**Systematic Errors Detected:**
- Positive words misclassified as negative: {pos_as_neg} ({pos_as_neg/len(df)*100:.1f}%)
- Negative words misclassified as positive: {neg_as_pos} ({neg_as_pos/len(df)*100:.1f}%)

These patterns suggest potential cultural bias where {language} emotional concepts
are being mapped incorrectly to English sentiment categories.

## Detailed Error Analysis

Total errors: {len(df[~df['agreement']])}

### Most Common Error Patterns:
"""
        
        errors_df = df[~df['agreement']]
        if len(errors_df) > 0:
            error_patterns = errors_df.groupby([validated_col, 'baseline_prediction']).size().reset_index(name='count')
            error_patterns = error_patterns.sort_values('count', ascending=False).head(5)
            
            for _, row in error_patterns.iterrows():
                report += f"- {row[validated_col]} → {row['baseline_prediction']}: {row['count']} occurrences\n"
        
        report += f"""

## Implications

### For AI Development
1. Current systems show {100-accuracy:.1f}% error rate on {language}
2. Systematic biases indicate training data gaps
3. Cultural concepts not adequately represented

### For Research
1. Document "cultural sentiment gap"
2. Build corpus of culture-specific emotions
3. Develop better multilingual models

### Recommendations
1. Include African language speakers in model development
2. Validate sentiment datasets with native speakers
3. Consider cultural context in sentiment classification
4. Build Ubuntu-aware sentiment models

## Methodology

- **Validation:** Native speaker annotations
- **Baseline:** {baseline} zero-shot classification
- **Metrics:** Accuracy, confusion matrix, error patterns
- **Language:** {language}

## Data Availability

All data, including validated lexicon and baseline predictions, available for research purposes.

---

Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Display report
        st.markdown(report)
        
        st.divider()
        
        # Export options
        col_rep1, col_rep2, col_rep3 = st.columns(3)
        
        with col_rep1:
            st.download_button(
                "📄 Download Report (TXT)",
                report,
                f"bias_audit_{baseline}_{language}.txt",
                "text/plain",
                use_container_width=True
            )
        
        with col_rep2:
            # Full data export
            export_df = df.copy()
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                "📊 Download Data (CSV)",
                csv_data,
                f"bias_data_{baseline}_{language}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col_rep3:
            # Research paper format
            paper_format = f"""% Bias Audit: {baseline} on {language}
% Auto-generated research data

\\section{{Results}}

Overall accuracy: {accuracy:.1f}\\%

\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lcc}}
\\hline
Sentiment & Accuracy & Count \\\\
\\hline
"""
            for sent, acc in sentiment_accuracy.items():
                count = len(df[df[validated_col] == sent])
                paper_format += f"{sent.title()} & {acc:.1f}\\% & {count} \\\\\n"
            
            paper_format += """\\hline
\\end{tabular}
\\caption{Performance by sentiment category}
\\end{table}
"""
            
            st.download_button(
                "📝 LaTeX Format",
                paper_format,
                f"bias_audit_{baseline}_{language}.tex",
                "text/plain",
                use_container_width=True
            )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p><strong>🔬 Bias Detection Lab</strong></p>
    <p>Auditing AI systems for cultural and linguistic bias</p>
    <p><em>Making bias visible, quantifiable, and actionable</em></p>
</div>
""", unsafe_allow_html=True)