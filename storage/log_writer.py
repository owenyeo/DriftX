import json
from datetime import datetime
from pathlib import Path

from database.db import SessionLocal
from database.models import InferenceLog
from datetime import datetime

async def log_inference_to_db(payload: dict):

    # you have this statement to ensure that session is deleted and not zombie
    async with SessionLocal() as session:
        log = InferenceLog(
            input_data=payload.get("input"),
            output_data=payload.get("output"),
            latency=payload["latency"],
            status_code=payload["status_code"],
            timestamp=datetime.fromisoformat(payload["timestamp"])
        )
        session.add(log)
        await session.commit()


# Path to the log file (relative to project root)
log_file = Path("storage/inference_logs.jsonl")

# Ensure the directory exists
log_file.parent.mkdir(parents=True, exist_ok=True)

def log_inference(entry: dict):
    """
    Appends an inference log to the JSONL log file.

    Args:
        entry (dict): A dictionary with keys like 'input', 'latency', 'status_code'
    """
    entry["timestamp"] = datetime.utcnow().isoformat()

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")