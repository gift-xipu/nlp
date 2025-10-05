"""
Main entry point for the African Languages Sentiment Analysis application.
Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path

# Configure the main page
st.set_page_config(
    page_title="African Languages Sentiment Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': """
        # African Languages Sentiment Analysis
        
        A comprehensive platform for generating sentiment lexicons for 
        Sepedi, Sesotho, and Setswana using Large Language Models.
        
        **Version:** 1.0.0
        """
    }
)

# Import custom CSS
from gui.styles.custom_css import apply_custom_css
apply_custom_css()

# Main content
st.title("🌍 African Languages Sentiment Analysis")
st.markdown("""
### Welcome to the Research Platform

This comprehensive system enables researchers to generate, analyze, and validate sentiment lexicons 
for **Sepedi**, **Sesotho**, and **Setswana** using state-of-the-art Large Language Models.
""")

st.divider()

# Feature overview
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 📝 Core Features
    
    - **List Words Generation**: Generate 1000+ sentiment-bearing words using LLMs
    - **Sentiment Bearing Analysis**: Label words with sentiment, confidence, and reasoning
    - **Lexicon Bootstrapping**: Expand lexicons using semantic similarity
    - **Corpus Management**: Upload, clean, and translate text corpora
    - **Validation Tools**: Automated and manual lexicon validation
    - **Analytics Dashboard**: Compare and analyze lexicon quality
    """)

with col2:
    st.markdown("""
    #### 🎯 Research Contributions
    
    - **Low-Resource NLP**: Addresses data scarcity in African languages
    - **Prompt Engineering**: Systematic evaluation of prompting strategies
    - **Scalable Methods**: Reduces reliance on manual annotation
    - **Quality Assessment**: Comprehensive validation framework
    - **Multilingual Support**: Covers three South African languages
    - **Open Research**: Reproducible methods and transparent processes
    """)

st.divider()

# Quick start guide
st.markdown("### 🚀 Quick Start Guide")

with st.expander("1️⃣ Generate Word Lists", expanded=True):
    st.markdown("""
    **Goal:** Generate 1000 sentiment words in your target language
    
    1. Navigate to **📝 List Words** page
    2. Configure your LLM (OpenAI, Claude, or Gemini)
    3. Select language and sentiment type
    4. Choose prompt strategy (few-shot recommended)
    5. Click **Generate** and export results
    
    💡 **Tip:** Use few-shot prompting for better quality
    """)

with st.expander("2️⃣ Label Sentiment Bearing"):
    st.markdown("""
    **Goal:** Analyze and label words with sentiment scores
    
    1. Navigate to **🎯 Sentiment Bearing** page
    2. Load your generated words or upload a file
    3. Configure analysis settings
    4. Run analysis to get sentiment labels, scores, and reasoning
    5. Export labeled dataset
    
    💡 **Tip:** Lower temperature (0.2-0.4) gives more consistent labels
    """)

with st.expander("3️⃣ Bootstrap Lexicon"):
    st.markdown("""
    **Goal:** Expand lexicon using semantic similarity
    
    1. Navigate to **🔄 Lexicon Bootstrapping** page
    2. Define seed words (high-confidence words)
    3. Provide candidate words pool
    4. Configure similarity threshold
    5. Run bootstrapping algorithm
    6. Review and export expanded lexicon
    
    💡 **Tip:** Start with similarity threshold of 0.7 and adjust
    """)

with st.expander("4️⃣ Manage Corpus"):
    st.markdown("""
    **Goal:** Process and clean text corpora
    
    1. Navigate to **📊 Corpus Manager** page
    2. Upload or paste corpus text
    3. Apply cleaning operations (remove URLs, stopwords, etc.)
    4. Use LLM for intelligent cleaning
    5. Translate if needed
    6. Export processed corpus
    
    💡 **Tip:** Clean before translating for better results
    """)

with st.expander("5️⃣ Validate Quality"):
    st.markdown("""
    **Goal:** Assess and validate lexicon quality
    
    1. Navigate to **✅ Validation** page
    2. Load lexicon to validate
    3. Run automatic validation
    4. Review quality metrics and recommendations
    5. Optionally perform manual native speaker review
    6. Export quality report
    
    💡 **Tip:** Combine automatic and manual validation
    """)

with st.expander("6️⃣ Analyze Results"):
    st.markdown("""
    **Goal:** Compare lexicons and analyze patterns
    
    1. Navigate to **📈 Analytics** page
    2. Select analysis type
    3. Load lexicons to compare
    4. View distributions and statistics
    5. Export visualizations and insights
    
    💡 **Tip:** Compare different prompt strategies to optimize
    """)

st.divider()

# System architecture
st.markdown("### 🏗️ System Architecture")

col_arch1, col_arch2, col_arch3 = st.columns(3)

with col_arch1:
    st.markdown("""
    **LLM Integration**
    - OpenAI GPT-4o
    - Anthropic Claude Sonnet 4.5
    - Google Gemini 1.5 Pro
    
    Dynamic model selection with unified interface
    """)

with col_arch2:
    st.markdown("""
    **Prompt Strategies**
    - Zero-shot
    - Few-shot (Recommended)
    - In-context learning
    
    Systematic prompt engineering framework
    """)

with col_arch3:
    st.markdown("""
    **Languages**
    - Sepedi (Northern Sotho)
    - Sesotho (Southern Sotho)
    - Setswana (Tswana)
    
    Bantu language family support
    """)

st.divider()

# Workflow diagram
st.markdown("### 📊 Typical Workflow")

st.markdown("""
```
┌─────────────────┐
│  1. List Words  │──┐
│  (1000+ words)  │  │
└─────────────────┘  │
                     ↓
┌──────────────────────────┐
│  2. Sentiment Bearing    │
│  (Label + Score + Reason)│
└──────────────────────────┘
         │
         ↓
┌─────────────────────┐
│  3. Bootstrapping   │
│  (Expand Lexicon)   │
└─────────────────────┘
         │
         ↓
┌──────────────────┐
│  4. Validation   │
│  (Quality Check) │
└──────────────────┘
         │
         ↓
┌─────────────────┐
│  5. Analytics   │
│  (Compare/Eval) │
└─────────────────┘
```
""")

st.divider()

# Statistics
if any(key in st.session_state for key in ['generated_words', 'labeled_words', 'bootstrap_lexicon']):
    st.markdown("### 📊 Current Session Statistics")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        generated_count = len(st.session_state.get('generated_words', []))
        st.metric("Generated Words", generated_count)
    
    with col_stat2:
        labeled_count = len(st.session_state.get('labeled_words', []))
        st.metric("Labeled Words", labeled_count)
    
    with col_stat3:
        bootstrap_count = len(st.session_state.get('bootstrap_lexicon', []))
        st.metric("Bootstrap Lexicon", bootstrap_count)
    
    with col_stat4:
        corpus_length = len(st.session_state.get('corpus_text', ''))
        st.metric("Corpus Size", f"{corpus_length:,} chars")

# Research context
st.divider()
st.markdown("### 🔬 Research Context")

st.markdown("""
This platform supports research on **sentiment analysis for low-resource African languages**. 
Key research questions addressed:

1. **How can LLMs be adapted to generate sentiment lexicons for African languages?**
2. **What prompt engineering strategies are most effective?**
3. **How can we reduce reliance on manual annotation?**
4. **What quality metrics are appropriate for evaluating generated lexicons?**
5. **How can bootstrapping methods expand lexicons effectively?**

**Methodology:** Systematic evaluation of zero-shot, few-shot, and in-context prompting 
strategies across multiple LLM providers, with comprehensive validation frameworks.

**Impact:** Broadens inclusivity in multilingual NLP by providing scalable tools for 
under-resourced languages, advancing both practical applications and theoretical 
understanding of LLM capabilities.
""")

# Getting started
st.divider()
st.markdown("### 🎓 Getting Started")

col_start1, col_start2 = st.columns(2)

with col_start1:
    st.info("""
    **New Users:**
    
    1. Get API keys for your preferred LLM provider
    2. Start with the **List Words** page
    3. Generate a small batch (100 words) to test
    4. Review the results and adjust parameters
    5. Scale up to 1000+ words
    """)

with col_start2:
    st.success("""
    **Best Practices:**
    
    - Use **few-shot** prompting for quality
    - Validate results with native speakers
    - Compare multiple prompt strategies
    - Document your methodology
    - Export data at each stage
    """)

# Navigation
st.divider()
st.markdown("### 🧭 Navigate to:")

# Simplified navigation using markdown links instead of page_link
st.markdown("""
**Core Tasks:**
- 📝 **List Words** - Generate sentiment word lists (see sidebar)
- 🎯 **Sentiment Bearing** - Label words with scores and reasoning (see sidebar)
- 🔄 **Lexicon Bootstrapping** - Expand lexicons using semantic similarity (see sidebar)

**Processing & Analysis:**
- 📊 **Corpus Manager** - Upload, clean, and translate corpora (see sidebar)
- ✅ **Validation** - Automated and manual quality checks (see sidebar)
- 📈 **Analytics** - Compare and visualize results (see sidebar)

**Resources:**
- [Documentation](#)
- [GitHub Repo](#)
- [Setup Guide](#)

👈 **Use the sidebar to navigate between pages**
""")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p><strong>African Languages Sentiment Analysis Platform</strong></p>
    <p>Research tool for Sepedi, Sesotho, and Setswana sentiment lexicon generation</p>
    <p>Powered by OpenAI, Anthropic, and Google AI</p>
    <p style='margin-top: 10px; font-size: 0.9em;'>
        Version 1.0.0 | Built with Streamlit | © 2025
    </p>
</div>
""", unsafe_allow_html=True)