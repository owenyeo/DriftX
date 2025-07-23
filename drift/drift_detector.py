import pandas as pd
import numpy as np
import json
from datetime import datetime
from sqlalchemy.future import select
from database.models import InferenceLog  # or wherever you defined it
from database.db import SessionLocal
import asyncio

async def get_logs(session, limit: int = 1000):
    result = await session.execute(
        select(InferenceLog).order_by(InferenceLog.timestamp.desc()).limit(limit)
    )
    return result.scalars().all()

def normalize_logs(logs):
    raw = [
        {
            "timestamp": log.timestamp,
            "latency": log.latency,
            "status_code": log.status_code,
            **log.input_data,
            **(log.output_data or {})
        }
        for log in logs
    ]
    df = pd.DataFrame(raw)
    return df

def unpack_feature_list(df):
    if "features" in df.columns:
        features_df = pd.DataFrame(df["features"].tolist())
        features_df.columns = [f"feature_{i}" for i in features_df.columns]
        df = pd.concat([df.drop(columns=["features"]), features_df], axis=1)
    return df

def read_logs(path="storage/inference_logs.jsonl"):
    """Load and normalize the inference logs."""
    with open(path, "r") as f:
        logs = [json.loads(line) for line in f]
    valid_logs = [log for log in logs if isinstance(log.get("input"), dict) and log["input"]]

    if not logs:
        raise ValueError("No logs found.")

    df = pd.DataFrame(valid_logs)
    df_input = pd.json_normalize(df["input"])
    df_output = pd.json_normalize(df["output"])
    df_input["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df_input, df_output


def calculate_psi(expected, actual, buckets=10):
    """Compute PSI for a single feature."""
    expected_perc, _ = np.histogram(expected, bins=buckets)
    actual_perc, _ = np.histogram(actual, bins=buckets)

    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)

    # Avoid division by 0
    psi = np.sum((actual_perc - expected_perc) * np.log((actual_perc + 1e-6) / (expected_perc + 1e-6)))
    return psi


async def detect_data_drift():
    """Main function to detect drift."""
    async with SessionLocal() as session:
        logs = await get_logs(session)
        df = normalize_logs(logs)
        df = unpack_feature_list(df)

    # Split into baseline (older half) and current (newer half)
    mid_point = len(df) // 2
    baseline_df = df.iloc[:mid_point]
    current_df = df.iloc[mid_point:]

    print(f"\nRunning drift detection — {datetime.now().isoformat()}")
    for feature in baseline_df.columns:
        if feature == "timestamp":
            continue
        try:
            psi_score = calculate_psi(baseline_df[feature], current_df[feature])
            print(f"Feature `{feature}` PSI: {psi_score:.3f} — ", end="")
            if psi_score < 0.1:
                print("No drift")
            elif psi_score < 0.2:
                print("Slight drift")
            else:
                print("Significant drift detected!")
        except Exception as e:
            print(f"Feature `{feature}` — Error: {e}")

def detect_model_drift():
    """Main function to detect drift."""
    _, df_output = read_logs()

    # Split into baseline (older half) and current (newer half)
    mid_point = len(df_output) // 2
    baseline_df = df_output.iloc[:mid_point]
    current_df = df_output.iloc[mid_point:]

    print(f"\nRunning drift detection — {datetime.now().isoformat()}")
    for feature in baseline_df.columns:
        if feature == "timestamp":
            continue
        try:
            psi_score = calculate_psi(baseline_df[feature], current_df[feature])
            print(f"Feature `{feature}` PSI: {psi_score:.3f} — ", end="")
            if psi_score < 0.1:
                print("No drift")
            elif psi_score < 0.2:
                print("Slight drift")
            else:
                print("Significant drift detected!")
        except Exception as e:
            print(f"Feature `{feature}` — Error: {e}")

def run_drift_detector():
    asyncio.run(detect_data_drift())


if __name__ == "__main__":
    detect_model_drift()
    detect_data_drift()
