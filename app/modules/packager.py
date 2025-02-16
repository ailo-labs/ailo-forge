import zipfile
import os

def create_package(generated_files: dict, output_path: str = "custom_llm_package.zip") -> str:
    """
    Package the generated files into a ZIP archive.
    
    generated_files: dict with keys like 'script', 'dockerfile', 'env_config', 'hyperparams'
    output_path: where to save the ZIP file.
    
    Returns the path to the ZIP archive.
    """
    with zipfile.ZipFile(output_path, 'w') as zipf:
        for filename, content in generated_files.items():
            # Write each file into the zip archive
            zipf.writestr(f"{filename}.txt", content)
    return output_path

