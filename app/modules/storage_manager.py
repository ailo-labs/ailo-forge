import json

def get_model_metadata(model_choice: str) -> dict:
    """
    Return metadata for the selected model.
    In production, this would query a database or a manifest file.
    For now, use the same configuration as in model_selection.
    """
    from app.modules.model_selection import AVAILABLE_MODELS
    metadata = AVAILABLE_MODELS.get(model_choice)
    if not metadata:
        raise ValueError(f"No metadata found for model {model_choice}")
    return metadata

def generate_download_script(model_metadata: dict) -> str:
    """
    Generate a script (bash or Python) that fetches the model file from cloud storage.
    """
    script = f"""#!/bin/bash
# Download the model file from cloud storage
curl -o {model_metadata['name']}.bin "{model_metadata['file_url']}"
# Verify checksum if needed (placeholder)
echo "Downloaded {model_metadata['name']} model."
"""
    return script

