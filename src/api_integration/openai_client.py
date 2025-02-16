# openai_client.py

import openai
from src.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def generate_code(prompt: str, model="text-davinci-003", max_tokens=1500) -> str:
    """
    Calls the OpenAI API to generate code based on the given prompt.
    Adjust model and parameters as needed.
    """
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    code = response.choices[0].text.strip()
    return code

