from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_generate_llm():
    # Example test: sending a minimal payload
    payload = {
        "description": "Create a model specialized in financial trading analysis with high factual rigor.",
        "model_choice": "Llama 3.3",
        "toggles": {
            "jailbreak": False,
            "creative_burst": True,
            "factual_rigor": True,
            "verbose": False,
            "domain_focus": "finance"
        }
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data

