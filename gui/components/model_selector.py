"""Reusable model selector component with fine-tuning support."""

import streamlit as st
from tasks.fine_tuning import FineTuningManager
from config.settings import MODELS_CONFIG
from typing import Tuple, Optional

def render_model_selector(
    provider_options: list = ["OpenAI", "Claude", "Gemini"],
    language: Optional[str] = None,
    task_type: str = 'all',
    key_prefix: str = ""
) -> Tuple[str, str, str, bool, Optional[str]]:
    """
    Render model selector with fine-tuning support.
    
    Args:
        provider_options: List of provider names
        language: Target language for fine-tuned models
        task_type: Type of task ('list_words', 'sentiment_bearing', 'translation', 'all')
        key_prefix: Unique prefix for Streamlit keys
    
    Returns:
        (provider_key, model, api_key, use_finetuned, finetuned_model_id)
    """
    manager = FineTuningManager()
    
    # Provider selection
    provider = st.selectbox(
        "LLM Provider",
        provider_options,
        help="Select language model provider",
        key=f"{key_prefix}_provider"
    )
    
    provider_key = provider.lower()
    available_models = MODELS_CONFIG[provider_key]['models']
    default_model = MODELS_CONFIG[provider_key]['default']
    
    # Check for fine-tuned models
    has_finetuned = False
    if language:
        has_finetuned = manager.has_finetuned_model(provider_key, language)
    
    # Fine-tuning checkbox
    use_finetuned = st.checkbox(
        "🎓 Use Fine-Tuned Model",
        value=False,
        disabled=not has_finetuned,
        help="Use your fine-tuned model" if has_finetuned else "No fine-tuned model available. Create one in Fine-Tuning page.",
        key=f"{key_prefix}_use_ft"
    )
    
    finetuned_model_id = None
    
    if use_finetuned and has_finetuned:
        # Get fine-tuned model
        finetuned_model_id = manager.get_model_for_language(provider_key, language, task_type)
        if not finetuned_model_id:
            finetuned_model_id = manager.get_model_for_language(provider_key, language, 'all')
        
        st.success("✅ Using fine-tuned model")
        st.code(finetuned_model_id, language=None)
        
        # Allow override
        with st.expander("🔧 Advanced"):
            custom_model = st.text_input(
                "Custom Model ID",
                value=finetuned_model_id,
                help="Override with different model ID",
                key=f"{key_prefix}_custom_model"
            )
            if custom_model != finetuned_model_id:
                finetuned_model_id = custom_model
        
        model = finetuned_model_id
    else:
        # Regular model selection
        model = st.selectbox(
            "Model",
            available_models,
            index=available_models.index(default_model) if default_model in available_models else 0,
            key=f"{key_prefix}_model"
        )
    
    # API Key
    api_key = st.text_input(
        "API Key",
        type="password",
        help="Enter your API key",
        key=f"{key_prefix}_api_key"
    )
    
    return provider_key, model, api_key, use_finetuned, finetuned_model_id