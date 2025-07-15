import json
from datetime import datetime
from pathlib import Path

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