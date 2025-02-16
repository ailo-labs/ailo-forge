from celery import Celery
from app.config import settings
from app.modules import code_generator, packager, storage_manager

# Initialize Celery
celery_app = Celery('tasks', broker=settings.REDIS_URL)

@celery_app.task(bind=True)
def generate_custom_llm_task(self, input_data: dict, selected_model: dict, toggles_config: dict) -> dict:
    """
    Celery task to generate the custom LLM code package.
    """
    try:
        # Step 1: Generate code files using OpenAI API
        generated_files = code_generator.generate_code(input_data, selected_model, toggles_config)
        
        # Step 2: Generate a download script for the model file
        download_script = storage_manager.generate_download_script(selected_model)
        generated_files['download_script'] = download_script
        
        # Step 3: Package all generated files into a ZIP archive
        package_path = packager.create_package(generated_files)
        
        return {"package_path": package_path, "status": "success"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

