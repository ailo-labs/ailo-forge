import os
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Get the MongoDB URI (ensure your URI includes a default database name)
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set")

# Create an asynchronous MongoDB client and get the default database
client = AsyncIOMotorClient(MONGO_URI)
db = client.get_default_database()
model_states_coll = db["model_states"]

# Create FastAPI app
app = FastAPI()

# Configure CORS to allow all origins (for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request models
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

@app.get("/")
async def read_root():
    return {"message": "Backend is running."}

@app.post("/modify")
async def modify_model(payload: dict):
    try:
        # Choose Standard or Advanced based on payload content
        if payload.get("advanced"):
            req = AdvancedModifyRequest(**payload)
            model_version = req.model_version
            data = {
                "model_version": model_version,
                "advanced": True,
                "modifications": req.modifications
            }
        else:
            req = StandardModifyRequest(**payload)
            model_version = req.model_version
            data = {
                "model_version": model_version,
                "instructions": req.instructions,
                "temperature": req.temperature,
                "responseSpeed": req.responseSpeed,
                "tokenLimit": req.tokenLimit,
                "unrestrictedMode": req.unrestrictedMode,
                "advanced": False
            }
        
        # Update or insert the model state into MongoDB
        result = await model_states_coll.update_one(
            {"model_version": model_version},
            {"$set": data},
            upsert=True
        )
        request_id = f"{model_version}_{'advanced' if payload.get('advanced') else 'standard'}"
        logging.info(f"Modified model {model_version}: {result.raw_result}")
        return {"request_id": request_id, "status": "ok"}
    except Exception as e:
        logging.exception("Error in /modify endpoint")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/run")
async def run_model(payload: dict):
    model_version = payload.get("model_version")
    prompt = payload.get("prompt", "")
    if not model_version or not prompt:
        raise HTTPException(status_code=400, detail="model_version and prompt are required")
    try:
        # Simulated inference: reverse the prompt as a placeholder response
        response_text = prompt[::-1]
        return {"success": True, "response": {"text": response_text}}
    except Exception as e:
        logging.exception("Error in /run endpoint")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/train")
async def train_model(
    trainingObjective: str = Form(...),
    dataset: list[UploadFile] = File(...)
):
    try:
        logging.info(f"Training Objective: {trainingObjective}")
        for file in dataset:
            contents = await file.read()
            logging.info(f"Received file {file.filename} ({len(contents)} bytes)")
        return {"message": "Training initiated successfully!"}
    except Exception as e:
        logging.exception("Error in /train endpoint")
        raise HTTPException(status_code=500, detail="Training failed.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
