# tasks/run_task.py
from models.openai_client import OpenAIClient
from prompts.loader import load_prompt

def run_task(model: str, prompt_style: str, language: str, user_input: str):
    # Load prompt template
    prompt_template = load_prompt(prompt_style, language)
    
    # Inject user input if needed
    prompt = prompt_template.replace("{input}", user_input)
    
    # Choose model dynamically
    if model == "openai":
        client = OpenAIClient(api_key="YOUR_KEY", model_name="gpt-4")
    elif model == "claude":
        from models.claude_client import ClaudeClient
        client = ClaudeClient(api_key="YOUR_KEY")
    elif model == "gemini":
        from models.gemini_client import GeminiClient
        client = GeminiClient(api_key="YOUR_KEY")
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Generate response
    return client.generate(prompt)
