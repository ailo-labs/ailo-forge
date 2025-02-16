import openai
from app.config import settings

# Configure the OpenAI API key
openai.api_key = settings.OPENAI_API_KEY

def generate_code(input_data: dict, selected_model: dict, toggles_config: dict) -> dict:
    """
    Generate code and configuration for the custom LLM package.
    This function calls the OpenAI API with a prompt that includes the user input,
    selected model metadata, and toggles configuration.
    
    Returns a dict containing generated script content, Dockerfile, and config files.
    """
    prompt = f"""
    Generate a complete set of deployment files for a custom LLM.
    The base model is {selected_model['name']} with file {selected_model['file_url']}.
    The model should be specialized for: {input_data['description']}.
    Apply these toggle settings: {toggles_config}.
    Include:
    - A Python script for model configuration and initialization.
    - A Dockerfile for containerized deployment.
    - Environment configuration files (in YAML or .env format).
    - Hyperparameter settings in JSON format.
    Provide clear comments in the code.
    """
    # Call OpenAI API to generate the code (simulate for now)
    response = openai.Completion.create(
        engine="text-davinci-003",  # adjust engine as needed
        prompt=prompt,
        max_tokens=1500,           # adjust max tokens
        temperature=0.7
    )
    generated_text = response.choices[0].text.strip()
    
    # For simplicity, we assume the generated_text is a JSON with keys: script, dockerfile, env_config, hyperparams.
    # In practice, you might need to parse or structure the output.
    try:
        generated_code = eval(generated_text)  # WARNING: eval() is used here for simplicity; use safe parsing in production.
    except Exception as e:
        raise ValueError("Failed to parse generated code: " + str(e))
    
    return generated_code

