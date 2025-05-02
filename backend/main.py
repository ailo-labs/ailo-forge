"""
main.py

A backend server for the Ailo Forge platform.
This FastAPI application exposes endpoints to:
  - Modify configuration files for LLM models.
  - Track progress of long-running modification/training tasks.
  - Provide a chat interface (using OpenAI’s GPT-4).
  - Trigger a training routine via the Hugging Face Trainer.

This code is designed for a production-quality application, with detailed comments and
progress reporting to guide users and developers alike.
"""

import os
import logging
import uuid
import asyncio
import json
from typing import List
from embeddings import compute_embedding

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Hugging Face Trainer and dataset imports for training demonstration
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# OpenAI for chat endpoint – using GPT-4 for updated responses
import openai

# ---------------------------
# Load environment variables and initialize logging
# ---------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Set OpenAI API key (ensure this key is for GPT‑4 access)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logging.warning("OPENAI_API_KEY not set. Chat endpoint will use fallback response.")

# ---------------------------
# MongoDB Connection
# ---------------------------
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set")
client = AsyncIOMotorClient(MONGO_URI)
db = client.get_default_database()
model_states_coll = db["model_states"]

# ---------------------------
# FastAPI App Setup
# ---------------------------
app = FastAPI(
    title="Ailo Forge Backend",
    description="Backend for LLM Modification, Training, and Chatting Platform.",
    version="1.0.0",
)

# Enable CORS for all origins – adjust for production as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# In-memory progress store for background tasks
# ---------------------------
progress_store = {}

# ---------------------------
# Pydantic Models for Modification Requests
# ---------------------------
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

# ---------------------------
# Root Endpoint
# ---------------------------
@app.get("/")
async def read_root():
    return {"message": "Ailo Forge backend is running."}

# ---------------------------
# File Modification Endpoint
# ---------------------------
@app.post("/modify-file")
async def modify_file(payload: dict, background_tasks: BackgroundTasks):
    """
    Reads the configuration file for the specified model (e.g., models/7B-Base/config.json),
    applies the modifications from the payload, writes the updated config back to disk, and
    starts a simulated long-running process to represent modification.
    """
    try:
        model_version = payload.get("model_version")
        modifications = payload.get("modifications")
        if not model_version or not modifications:
            raise HTTPException(status_code=400, detail="model_version and modifications are required")
        
        file_path = os.path.join("models", model_version, "config.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Configuration file for model {model_version} not found.")
        
        # Load and update the configuration file
        with open(file_path, "r") as f:
            config = json.load(f)
        config.update(modifications)
        with open(file_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Create a unique job ID and initialize progress tracking
        job_id = str(uuid.uuid4())
        progress_store[job_id] = {"percent": 0, "status": "in_progress"}
        # Start background task to simulate the modification process
        background_tasks.add_task(simulate_modification, job_id, 30)
        
        logging.info(f"Modified config for {model_version}. Job ID: {job_id}")
        return {"job_id": job_id, "status": "File modified; process started."}
    except Exception as e:
        logging.exception("Error in /modify-file endpoint")
        raise HTTPException(status_code=500, detail=str(e))

async def simulate_modification(job_id: str, total_time: int):
    """
    Simulates a long-running file modification process. Updates the progress_store each second.
    """
    for i in range(total_time):
        progress_store[job_id] = {"percent": int((i + 1) / total_time * 100), "status": "in_progress"}
        await asyncio.sleep(1)
    progress_store[job_id] = {"percent": 100, "status": "completed"}

# ---------------------------
# Progress Endpoint
# ---------------------------
@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    """
    Returns the current progress and status for a background job.
    """
    if job_id in progress_store:
        return progress_store[job_id]
    raise HTTPException(status_code=404, detail="Job not found.")

# ---------------------------
# Chat Endpoint Using OpenAI's GPT-4 API
# ---------------------------
@app.post("/run")
async def run_model(payload: dict):
    """
    Chat endpoint using OpenAI's ChatCompletion API with GPT-4. Expects a payload:
      {
        "model_version": "7B-Base",
        "prompt": "Hello, how are you?",
        "mode": "instruct" or "conversation" (optional)
      }
    If OpenAI API key is not set, falls back to a simple reversed prompt.
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
                "content": "You are an assistant that follows instructions precisely."
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful conversational assistant with updated knowledge."
            })
        messages.append({"role": "user", "content": prompt})
        
        if openai.api_key:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # use GPT‑4 for current information
                messages=messages,
                temperature=0.7,
                max_tokens=150,
            )
            response_text = response.choices[0].message.content.strip()
        else:
            response_text = prompt[::-1]  # fallback: reverse the prompt
        return {"success": True, "response": {"text": response_text}}
    except Exception as e:
        logging.exception("Error in /run endpoint")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# ---------------------------
# Training Endpoint (Demo with Hugging Face Trainer)
# ---------------------------
@app.post("/train")
async def train_model(
    trainingObjective: str = Form(...),
    dataset: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Accepts a training objective and a list of dataset files (text files are used for training; images are skipped).
    Launches a background training job using a Hugging Face Trainer demo (using 'distilgpt2' as a base model).
    """
    try:
        logging.info(f"Training Objective: {trainingObjective}")
        text_data = []
        for file in dataset:
            contents = await file.read()
            # Skip image files (basic signature check)
            if contents.startswith(b"\xFF\xD8") or contents.startswith(b"\x89PNG"):
                logging.info(f"Skipping image file: {file.filename}")
            else:
                text_data.append(contents.decode("utf-8", errors="ignore"))
        
        if not text_data:
            raise HTTPException(status_code=400, detail="No valid text data found for training.")
        
        job_id = str(uuid.uuid4())
        progress_store[job_id] = {"percent": 0, "status": "in_progress"}
        background_tasks.add_task(run_training_job, job_id, trainingObjective, text_data)
        return {"job_id": job_id, "message": "Training started. Check progress at /progress/{job_id}."}
    except Exception as e:
        logging.exception("Error in /train endpoint")
        raise HTTPException(status_code=500, detail="Training failed.")

def training_job(job_id: str, trainingObjective: str, text_data: List[str]):
    """
    Synchronous training job using Hugging Face Trainer.
    For a real application, replace this with your custom training pipeline.
    """
    try:
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create a dataset from the text data
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
        
        # Simulate training for 10 steps with progress updates
        import time
        for step in range(10):
            trainer.train(resume_from_checkpoint=None)
            progress_store[job_id] = {"percent": int((step + 1) / 10 * 100), "status": "in_progress"}
            time.sleep(1)
        progress_store[job_id] = {"percent": 100, "status": "completed"}
    except Exception as e:
        logging.exception("Error during training_job")
        progress_store[job_id] = {"percent": 0, "status": "failed"}

async def run_training_job(job_id: str, trainingObjective: str, text_data: List[str]):
    """
    Wraps the synchronous training job in an asynchronous thread.
    """
    try:
        await asyncio.to_thread(training_job, job_id, trainingObjective, text_data)
    except Exception as e:
        logging.exception("Error in run_training_job")
        progress_store[job_id] = {"percent": 0, "status": "failed"}

# ---------------------------
# Run the Application
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
