"""
Validation Dashboard - Gamified Interface for Native Speakers

Makes lexicon validation fun, engaging, and systematic.
Tracks progress, inter-annotator agreement, and quality metrics.

Research Innovation: Community-driven validation at scale
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from collections import Counter
import plotly.graph_objects as go

from gui.styles.custom_css import apply_custom_css, styled_header
from config.languages import get_supported_languages

st.set_page_config(
    page_title="Validation Dashboard",
    page_icon="✅",
    layout="wide"
)

apply_custom_css()

# Initialize session state
if 'validator_name' not in st.session_state:
    st.session_state.validator_name = ""
if 'validations' not in st.session_state:
    st.session_state.validations = []
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'session_stats' not in st.session_state:
    st.session_state.session_stats = {
        'validated': 0,
        'agreements': 0,
        'time_started': None
    }

# Load validation data
validation_file = Path('data/validations/validations.jsonl')
validation_file.parent.mkdir(parents=True, exist_ok=True)

def load_validations():
    """Load all validations from file."""
    if not validation_file.exists():
        return []
    
    validations = []
    with open(validation_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                validations.append(json.loads(line))
    return validations

def save_validation(validation_data):
    """Append validation to file."""
    with open(validation_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(validation_data, ensure_ascii=False) + '\n')

def calculate_validator_stats(validator_name):
    """Calculate stats for a specific validator."""
    validations = load_validations()
    user_validations = [v for v in validations if v.get('validator') == validator_name]
    
    if not user_validations:
        return None
    
    return {
        'total': len(user_validations),
        'languages': len(set(v['language'] for v in user_validations)),
        'avg_confidence': sum(v['confidence'] for v in user_validations) / len(user_validations),
        'date_started': min(v['timestamp'] for v in user_validations),
        'last_active': max(v['timestamp'] for v in user_validations)
    }

def get_leaderboard():
    """Get top validators."""
    validations = load_validations()
    
    if not validations:
        return []
    
    validator_counts = Counter(v.get('validator', 'Anonymous') for v in validations)
    return validator_counts.most_common(10)

# Header
st.title("✅ Validation Dashboard")
st.markdown("""
**Help Validate Sentiment Lexicons for African Languages**

Your expertise as a native speaker is invaluable! Validate word sentiments,
earn achievements, and contribute to groundbreaking research.
""")

st.divider()

# Check if user is logged in
if not st.session_state.validator_name:
    # Login screen
    st.markdown("### 👤 Welcome, Validator!")
    
    col_login1, col_login2 = st.columns([2, 1])
    
    with col_login1:
        name = st.text_input(
            "Your Name or Nickname",
            placeholder="e.g., Thabo M.",
            help="This will be used to track your contributions"
        )
        
        if st.button("🚀 Start Validating", type="primary", disabled=not name):
            st.session_state.validator_name = name
            st.session_state.session_stats['time_started'] = datetime.now().isoformat()
            st.rerun()
    
    with col_login2:
        st.info("""
        **Why Validate?**
        
        ✅ Help improve AI for African languages
        🏆 Earn achievements
        📊 See your impact
        🌍 Join the community
        """)
    
    st.divider()
    
    # Show leaderboard
    st.markdown("### 🏆 Top Validators")
    
    leaderboard = get_leaderboard()
    
    if leaderboard:
        for i, (name, count) in enumerate(leaderboard[:5], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "⭐"
            st.markdown(f"{emoji} **{name}** - {count} validations")
    else:
        st.info("Be the first validator!")

else:
    # Main validation interface
    
    # Sidebar - User stats
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.validator_name}")
        
        user_stats = calculate_validator_stats(st.session_state.validator_name)
        
        if user_stats:
            st.metric("Total Validated", user_stats['total'])
            st.metric("Languages", user_stats['languages'])
            st.metric("Avg Confidence", f"{user_stats['avg_confidence']:.2f}")
        
        st.divider()
        
        # Session stats
        st.markdown("### 📊 This Session")
        st.metric("Validated", st.session_state.session_stats['validated'])
        
        st.divider()
        
        if st.button("🚪 Logout"):
            st.session_state.validator_name = ""
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "✅ Validate",
        "🏆 Achievements", 
        "📊 My Stats",
        "🌍 Community"
    ])
    
    # TAB 1: Validate
    with tab1:
        styled_header("Validate Words")
        
        # Load words to validate
        if not st.session_state.current_batch:
            col_load1, col_load2 = st.columns([2, 1])
            
            with col_load1:
                language = st.selectbox("Select Language", get_supported_languages())
                
                uploaded = st.file_uploader(
                    "Upload Words to Validate (CSV/JSON)",
                    type=['csv', 'json']
                )
                
                if uploaded:
                    try:
                        if uploaded.name.endswith('.csv'):
                            df = pd.read_csv(uploaded)
                            words = df.to_dict('records')
                        else:
                            words = json.loads(uploaded.read())
                        
                        st.session_state.current_batch = words
                        st.session_state.current_index = 0
                        st.session_state.batch_language = language
                        st.success(f"✅ Loaded {len(words)} words")
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            
            with col_load2:
                st.info("""
                **Quick Start:**
                
                1. Select language
                2. Upload word list
                3. Start validating
                4. Earn achievements!
                """)
        
        else:
            # Validation interface
            batch = st.session_state.current_batch
            idx = st.session_state.current_index
            language = st.session_state.batch_language
            
            if idx >= len(batch):
                # Batch complete
                st.success(f"🎉 Batch Complete! You validated {len(batch)} words!")
                
                if st.button("🔄 Start New Batch"):
                    st.session_state.current_batch = []
                    st.session_state.current_index = 0
                    st.rerun()
            
            else:
                # Current word
                word_data = batch[idx]
                word = word_data.get('word', '')
                translation = word_data.get('translation', '')
                
                # Progress bar
                progress = (idx + 1) / len(batch)
                st.progress(progress)
                st.markdown(f"**Progress:** {idx + 1} / {len(batch)} ({progress*100:.0f}%)")
                
                st.divider()
                
                # Word display
                col_word1, col_word2 = st.columns([1, 2])
                
                with col_word1:
                    st.markdown("### 📝 Word")
                    st.markdown(f"# {word}")
                    st.markdown(f"*{translation}*")
                
                with col_word2:
                    # Show existing sentiment if available
                    if 'sentiment' in word_data:
                        st.markdown("### 🤖 LLM Prediction")
                        st.info(f"Sentiment: **{word_data['sentiment']}**")
                        if 'confidence_score' in word_data:
                            st.info(f"Confidence: {word_data['confidence_score']:.2f}")
                        if 'reasoning' in word_data:
                            with st.expander("🧠 Reasoning"):
                                st.write(word_data['reasoning'])
                
                st.divider()
                
                # Validation form
                st.markdown("### ✅ Your Validation")
                
                col_val1, col_val2 = st.columns([2, 1])
                
                with col_val1:
                    # Sentiment selection
                    sentiment_options = ['positive', 'negative', 'neutral', 'not_sure']
                    sentiment_labels = {
                        'positive': '😊 Positive',
                        'negative': '😔 Negative',
                        'neutral': '😐 Neutral',
                        'not_sure': '🤔 Not Sure'
                    }
                    
                    selected_sentiment = st.radio(
                        "Select Sentiment",
                        sentiment_options,
                        format_func=lambda x: sentiment_labels[x],
                        horizontal=True,
                        key=f"sentiment_{idx}"
                    )
                
                with col_val2:
                    confidence = st.slider(
                        "How confident are you?",
                        0.0, 1.0, 0.8, 0.1,
                        help="1.0 = Very confident, 0.0 = Just guessing",
                        key=f"confidence_{idx}"
                    )
                
                # Cultural notes
                cultural_note = st.text_area(
                    "Cultural Context (Optional)",
                    placeholder="Add any cultural context, usage notes, or examples...",
                    key=f"cultural_{idx}",
                    height=100
                )
                
                # Example usage
                example = st.text_input(
                    "Example Sentence (Optional)",
                    placeholder="Show how this word is used in context...",
                    key=f"example_{idx}"
                )
                
                st.divider()
                
                # Action buttons
                col_action1, col_action2, col_action3 = st.columns([1, 1, 1])
                
                with col_action1:
                    if st.button("⏭️ Skip", use_container_width=True):
                        st.session_state.current_index += 1
                        st.rerun()
                
                with col_action2:
                    if st.button("🚩 Report Issue", use_container_width=True):
                        # Save as issue
                        issue_data = {
                            'word': word,
                            'translation': translation,
                            'language': language,
                            'validator': st.session_state.validator_name,
                            'timestamp': datetime.now().isoformat(),
                            'issue_type': 'reported'
                        }
                        save_validation(issue_data)
                        st.session_state.current_index += 1
                        st.success("✅ Issue reported")
                        st.rerun()
                
                with col_action3:
                    if st.button("✅ Submit & Next", type="primary", use_container_width=True):
                        # Save validation
                        validation_data = {
                            'word': word,
                            'translation': translation,
                            'language': language,
                            'validator': st.session_state.validator_name,
                            'validated_sentiment': selected_sentiment,
                            'confidence': confidence,
                            'cultural_note': cultural_note if cultural_note else None,
                            'example': example if example else None,
                            'llm_sentiment': word_data.get('sentiment'),
                            'agreement': selected_sentiment == word_data.get('sentiment'),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        save_validation(validation_data)
                        
                        # Update session stats
                        st.session_state.session_stats['validated'] += 1
                        if validation_data['agreement']:
                            st.session_state.session_stats['agreements'] += 1
                        
                        st.session_state.current_index += 1
                        st.success("✅ Validated!")
                        st.rerun()
    
    # TAB 2: Achievements
    with tab2:
        styled_header("🏆 Your Achievements")
        
        user_stats = calculate_validator_stats(st.session_state.validator_name)
        
        if not user_stats:
            st.info("Start validating to earn achievements!")
        else:
            total = user_stats['total']
            
            # Achievement badges
            achievements = []
            
            if total >= 10:
                achievements.append(("🌱", "Seedling", "Validated 10 words"))
            if total >= 50:
                achievements.append(("🌿", "Sprout", "Validated 50 words"))
            if total >= 100:
                achievements.append(("🌳", "Tree", "Validated 100 words"))
            if total >= 500:
                achievements.append(("🏆", "Master Validator", "Validated 500 words"))
            if total >= 1000:
                achievements.append(("👑", "Legend", "Validated 1000 words"))
            
            if user_stats['languages'] >= 2:
                achievements.append(("🌍", "Multilingual", "Validated 2+ languages"))
            
            if user_stats['avg_confidence'] >= 0.9:
                achievements.append(("💎", "Confident", "Avg confidence 0.9+"))
            
            # Display achievements
            cols = st.columns(3)
            for i, (emoji, title, desc) in enumerate(achievements):
                with cols[i % 3]:
                    st.markdown(f"### {emoji} {title}")
                    st.info(desc)
            
            # Next achievement
            st.divider()
            st.markdown("### 🎯 Next Milestone")
            
            next_milestones = [10, 50, 100, 500, 1000]
            next_target = next((m for m in next_milestones if m > total), 1000)
            
            progress_to_next = total / next_target
            st.progress(progress_to_next)
            st.markdown(f"**{total} / {next_target}** words validated")
    
    # TAB 3: My Stats
    with tab3:
        styled_header("📊 My Statistics")
        
        validations = load_validations()
        my_validations = [v for v in validations if v.get('validator') == st.session_state.validator_name]
        
        if not my_validations:
            st.info("No validations yet. Start validating!")
        else:
            # Summary stats
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("Total Validated", len(my_validations))
            
            with col_stat2:
                agreements = sum(1 for v in my_validations if v.get('agreement', False))
                agreement_rate = agreements / len(my_validations) * 100
                st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
            
            with col_stat3:
                avg_conf = sum(v['confidence'] for v in my_validations) / len(my_validations)
                st.metric("Avg Confidence", f"{avg_conf:.2f}")
            
            with col_stat4:
                languages = len(set(v['language'] for v in my_validations))
                st.metric("Languages", languages)
            
            # Sentiment distribution
            st.divider()
            st.markdown("### 📊 Your Sentiment Distribution")
            
            sentiment_counts = Counter(v['validated_sentiment'] for v in my_validations)
            
            fig = go.Figure(data=[go.Pie(
                labels=list(sentiment_counts.keys()),
                values=list(sentiment_counts.values()),
                hole=0.3
            )])
            
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Community
    with tab4:
        styled_header("🌍 Community Dashboard")
        
        validations = load_validations()
        
        col_comm1, col_comm2 = st.columns([2, 1])
        
        with col_comm1:
            st.markdown("### 🏆 Leaderboard")
            
            leaderboard = get_leaderboard()
            
            for i, (name, count) in enumerate(leaderboard, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                is_you = name == st.session_state.validator_name
                
                if is_you:
                    st.markdown(f"**{emoji} {name} (You!) - {count} validations ⭐**")
                else:
                    st.markdown(f"{emoji} {name} - {count} validations")
        
        with col_comm2:
            st.markdown("### 📊 Community Stats")
            
            st.metric("Total Validations", len(validations))
            
            validators = len(set(v.get('validator', 'Anonymous') for v in validations))
            st.metric("Active Validators", validators)
            
            languages = len(set(v['language'] for v in validations))
            st.metric("Languages Covered", languages)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p><strong>✅ Thank You for Contributing!</strong></p>
    <p>Your validations help build better AI for African languages</p>
    <p><em>Every validation counts toward groundbreaking research</em></p>
</div>
""", unsafe_allow_html=True)