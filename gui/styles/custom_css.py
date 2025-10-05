"""
Clean, minimal styling for African Languages Sentiment Analysis.
Uses Streamlit's default dark theme with minor enhancements.
"""

import streamlit as st

def apply_custom_css():
    """
    Apply minimal custom styling.
    Call this function after st.set_page_config() on each page.
    """
    st.markdown("""
    <style>
    /* Minimal tweaks only */
    
    /* Slightly rounded buttons */
    .stButton button {
        border-radius: 5px;
    }
    
    /* Rounded dataframes */
    .stDataFrame {
        border-radius: 5px;
    }
    
    /* Clean section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #444;
    }
    </style>
    """, unsafe_allow_html=True)

def styled_header(title: str):
    """
    Simple section header.
    
    Args:
        title: Header text
    """
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)