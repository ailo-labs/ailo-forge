# backend/main.py
import os
import logging
import uuid
import asyncio
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from typing import List

# For training demonstration using Hugging Face Transformers and Datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.warning("OPENAI_API_KEY not set. Using fallback responses for /run endpoint.")

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set")
client = AsyncIOMotorClient(MONGO_URI)
db = client.get_default_database()
model_states_coll = db["model_states"]

app = FastAPI()

# Enable CORS for all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory progress store
progress_store = {}

# ------------------------------
# Pydantic Models
# ------------------------------

class StandardModifyRequest(BaseModel):
    model_version: str
    instructions: str = ""
    temperature: float = 0.5
    responseSpeed: int = 50
    tokenLimit: int = 512
    unrestrictedMode: bool = False

class AdvancedModifyRequest(BaseModel):
    model_version: str
    advanced: bool
    modifications: dict

# ------------------------------
# Helper Functions
# ------------------------------

async def simulate_modification(job_id: str, total_time: int):
    """Simulates a long running modification process with cancel support."""
    logger.info(f"Starting simulation for job {job_id} for {total_time} seconds.")
    for i in range(total_time):
        if progress_store[job_id].get("cancel_requested"):
            progress_store[job_id] = {"percent": int((i/total_time)*100), "status": "canceled"}
            logger.info(f"Job {job_id} was canceled at {i} seconds.")
            return
        progress_store[job_id] = {
            "percent": int(((i + 1) / total_time) * 100),
            "status": "in_progress",
            "cancel_requested": False
        }
        await asyncio.sleep(1)
    progress_store[job_id] = {"percent": 100, "status": "completed"}
    logger.info(f"Job {job_id} completed modification simulation.")

def training_job(job_id: str, trainingObjective: str, text_data: List[str]):
    """Synchronous training job simulation."""
    try:
        logger.info(f"Starting training job {job_id} with objective: {trainingObjective}")
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create dataset from text data
        ds = Dataset.from_dict({"text": text_data})
        
        def tokenize_fn(examples):
            tokenized = tokenizer(examples["text"], truncation=True, max_length=128)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        ds = ds.map(tokenize_fn, batched=True)
        ds = ds.remove_columns(["text"])
        ds.set_format("torch")
        
        training_args = TrainingArguments(
            output_dir="trained_model",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            logging_steps=1,
            logging_dir="./logs",
            save_steps=500,
            disable_tqdm=True,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds,
        )
        
        import time
        total_steps = 10
        for step in range(total_steps):
            if progress_store[job_id].get("cancel_requested"):
                progress_store[job_id] = {"percent": int((step/total_steps)*100), "status": "canceled"}
                logger.info(f"Training job {job_id} was canceled at step {step}.")
                return
            trainer.train(resume_from_checkpoint=None)
            progress_store[job_id] = {"percent": int(((step + 1)/total_steps)*100), "status": "in_progress", "cancel_requested": False}
            time.sleep(1)
        
        progress_store[job_id] = {"percent": 100, "status": "completed"}
        logger.info(f"Training job {job_id} completed successfully.")
    except Exception as e:
        logger.exception("Error in training job:")
        progress_store[job_id] = {"percent": 0, "status": "failed"}

async def run_training_job(job_id: str, trainingObjective: str, text_data: List[str]):
    """Runs the synchronous training job in a separate thread."""
    try:
        await asyncio.to_thread(training_job, job_id, trainingObjective, text_data)
    except Exception as e:
        logger.exception("Error running training job in thread:")
        progress_store[job_id] = {"percent": 0, "status": "failed"}

# ------------------------------
# Endpoints
# ------------------------------

@app.get("/")
async def read_root():
    """Root endpoint."""
    return {"message": "Backend is running."}

@app.post("/modify-file")
async def modify_file(payload: dict, background_tasks: BackgroundTasks):
    """Modify file endpoint."""
    try:
        model_version = payload.get("model_version")
        modifications = payload.get("modifications")
        if not model_version or not modifications:
            raise HTTPException(status_code=400, detail="model_version and modifications are required")
        
        file_path = os.path.join("models", f"{model_version}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Configuration file for model {model_version} not found.")
        
        # Load current configuration
        with open(file_path, "r") as f:
            config = json.load(f)
        
        # Update configuration
        config.update(modifications)
        
        # Save updated configuration
        with open(file_path, "w") as f:
            json.dump(config, f, indent=2)
        
        job_id = str(uuid.uuid4())
        progress_store[job_id] = {"percent": 0, "status": "in_progress", "cancel_requested": False}
        background_tasks.add_task(simulate_modification, job_id, 30)
        
        logger.info(f"Modified config for model {model_version} at {file_path}, job_id: {job_id}")
        return {"job_id": job_id, "status": "file modified and process started"}
    except Exception as e:
        logger.exception("Error in /modify-file endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    """Return progress information for a given job."""
    if job_id in progress_store:
        return progress_store[job_id]
    raise HTTPException(status_code=404, detail="Job not found")

@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    if job_id not in progress_store:
        raise HTTPException(status_code=404, detail="Job not found")
    progress_store[job_id]["cancel_requested"] = True
    logger.info(f"Cancel requested for job {job_id}")
    return {"job_id": job_id, "status": "cancel_requested"}

@app.post("/run")
async def run_model(payload: dict):
    """
    Chat endpoint: Uses GPT-4 to generate responses.
    Expected payload:
    {
        "model_version": "7B-Base",
        "prompt": "Hello, how are you?",
        "mode": "instruct" or "conversation"
    }
    """
    model_version = payload.get("model_version")
    prompt = payload.get("prompt", "")
    chat_mode = payload.get("mode", "conversation")
    if not model_version or not prompt:
        raise HTTPException(status_code=400, detail="model_version and prompt are required")
    try:
        messages = []
        if chat_mode == "instruct":
            messages.append({
                "role": "system",
                "content": "You are an assistant that follows instructions precisely. Provide detailed, candid, and up-to-date responses."
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful conversational assistant with current information. Provide clear and direct answers."
            })
        messages.append({"role": "user", "content": prompt})
        
        # Use GPT-4 if available; fallback to reversing prompt if API key is missing
        if openai.api_key:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=150,
            )
            response_text = response.choices[0].message.content.strip()
        else:
            response_text = prompt[::-1]
        return {"success": True, "response": {"text": response_text}}
    except Exception as e:
        logger.exception("Error in /run endpoint")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/train")
async def train_model(
    trainingObjective: str = Form(...),
    dataset: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Training endpoint: Accepts training objective and dataset files.
    For text files, content is used for training; image files are skipped.
    """
    try:
        logger.info(f"Training objective: {trainingObjective}")
        text_data = []
        for file in dataset:
            contents = await file.read()
            if contents.startswith(b"\xFF\xD8") or contents.startswith(b"\x89PNG"):
                logger.info(f"Skipping image file: {file.filename}")
            else:
                text_data.append(contents.decode("utf-8", errors="ignore"))
        
        if not text_data:
            raise HTTPException(status_code=400, detail="No valid text data found for training.")
        
        job_id = str(uuid.uuid4())
        progress_store[job_id] = {"percent": 0, "status": "in_progress", "cancel_requested": False}
        background_tasks.add_task(run_training_job, job_id, trainingObjective, text_data)
        return {"job_id": job_id, "message": "Training started. Poll progress at /progress/{job_id}."}
    except Exception as e:
        logger.exception("Error in /train endpoint")
        raise HTTPException(status_code=500, detail="Training failed.")
