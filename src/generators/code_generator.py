# code_generator.py

from src.api_integration.openai_client import generate_code
from src.toggles import apply_toggle_settings

def create_custom_llm_code(user_description: str, toggles: dict, base_model: str) -> str:
    """
    Generate the custom LLM code package based on the user description,
    toggle settings, and selected base model.
    """
    # Base prompt template
    prompt = f"""
You are an expert AI developer.
Generate code that creates a custom large language model based on the following description:
"{user_description}"
Use the base model: {base_model}.
Include necessary hyperparameters, environment configuration, and a Dockerfile setup.
    """
    # Apply toggle settings to refine output behavior
    prompt = apply_toggle_settings(prompt, toggles)
    
    # Use OpenAI API to generate code
    code_output = generate_code(prompt)
    return code_output

