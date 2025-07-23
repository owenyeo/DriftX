# FastAPI entrypoint
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
from api.router import router as api_router
import time
from datetime import datetime
from storage.log_writer import log_inference_to_db
from drift.drift_detector import detect_data_drift

from apscheduler.schedulers.background import BackgroundScheduler
from database.db import init_db

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await init_db()

@app.middleware("http")
async def log_and_forward(request: Request, call_next):
    body = await request.json()
    start_time = time.time()

    try:
        # Forward to actual model
        async with httpx.AsyncClient() as client:
            model_response = await client.post("http://localhost:9000/predict", json=body)
            model_response.raise_for_status()
            prediction = model_response.json()
    except Exception as e:
        prediction = {"error": str(e)}

    duration = time.time() - start_time

    await log_inference_to_db({
        "input": body,
        "output": prediction,
        "latency": duration,
        "status_code": model_response.status_code,
        "timestamp": datetime.now().isoformat()
    })

    return JSONResponse(content=prediction)

app.include_router(api_router)

scheduler = BackgroundScheduler()
scheduler.add_job(detect_data_drift, 'interval', hours=1)
scheduler.start()
