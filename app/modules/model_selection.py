import json

# In a real application, this might come from a database or external config.
AVAILABLE_MODELS = {
    "DeepSeek R1": {"name": "DeepSeek R1", "file_url": "https://your-cloud-storage/DeepSeekR1.bin", "size": "5GB"},
    "Llama 3.3": {"name": "Llama 3.3", "file_url": "https://your-cloud-storage/Llama3.3.bin", "size": "7GB"},
    "Bloom": {"name": "Bloom", "file_url": "https://your-cloud-storage/Bloom.bin", "size": "10GB"},
}

def select_model(model_choice: str) -> dict:
    """
    Return the configuration of the selected model.
    """
    model = AVAILABLE_MODELS.get(model_choice)
    if not model:
        raise ValueError(f"Model '{model_choice}' is not available.")
    return model

