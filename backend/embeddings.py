import numpy as np
import openai

def compute_embedding(text: str) -> np.ndarray:
    """
    Computes the embedding of the given text using OpenAI's Embedding API.
    Ensure your OPENAI_API_KEY is set in your environment.
    """
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"  # Example model name; adjust as needed
    )
    embedding = response["data"][0]["embedding"]
    return np.array(embedding, dtype='float32').reshape(1, -1)

