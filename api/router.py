from fastapi import APIRouter
from pydantic import BaseModel
import random
import time
import requests
from storage.log_writer import log_inference


# Create a new router object to register endpoints
router = APIRouter()

# Define the structure of the request body
class InferenceRequest(BaseModel):
    feature1: float
    feature2: float

# Define the /predict route that handles POST requests
@router.post("/predict")
async def predict(req: InferenceRequest):
    payload = req.dict()
    start_time = time.time()

    response = requests.post("http://localhost:9000/predict", json=payload)
    latency = time.time() - start_time

    log_inference({
        "input": payload,
        "output": response.json(),
        "latency": latency,
        "status_code": response.status_code
    })

    return response.json()
