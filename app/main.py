from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.modules import user_input, model_selection, toggle_manager, code_generator, packager, storage_manager
from app.tasks.generate_task import generate_custom_llm_task

app = FastAPI(title="Ailo Forge™", description="Custom LLM generation platform", version="0.1")

# Data model for user input
class LLMRequest(BaseModel):
    description: str
    model_choice: str  # e.g., "DeepSeek R1", "Llama 3.3", "Bloom"
    toggles: dict      # e.g., {"jailbreak": True, "creative_burst": False, ...}

@app.post("/generate")
async def generate_llm(request: LLMRequest):
    # Validate and process user input
    try:
        input_data = user_input.parse_input(request.description, request.toggles)
        selected_model = model_selection.select_model(request.model_choice)
        toggles_config = toggle_manager.process_toggles(request.toggles)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Trigger asynchronous task for code generation
    task = generate_custom_llm_task.delay(input_data, selected_model, toggles_config)
    return {"task_id": task.id, "status": "Task submitted. Check back for results."}

@app.get("/status/{task_id}")
async def task_status(task_id: str):
    # Endpoint to check the status of an asynchronous task.
    from celery.result import AsyncResult
    result = AsyncResult(task_id)
    return {"task_id": task_id, "status": result.status, "result": result.result}

