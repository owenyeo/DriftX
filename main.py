# FastAPI entrypoint
from fastapi import FastAPI, Request
from api.router import router as api_router
import time
from storage.log_writer import log_inference

app = FastAPI()

@app.middleware("http")
async def log_request_data(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    try:
        body = await request.json()
    except Exception:
        body = {}

    log_inference({
        "input": body,
        "latency": duration,
        "status_code": response.status_code
    })

    return response

app.include_router(api_router)
