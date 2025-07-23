import pandas as pd
import numpy as np
import json
from datetime import datetime


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


def detect_data_drift():
    """Main function to detect drift."""
    df_input, _ = read_logs()

    # Split into baseline (older half) and current (newer half)
    mid_point = len(df_input) // 2
    baseline_df = df_input.iloc[:mid_point]
    current_df = df_input.iloc[mid_point:]

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


if __name__ == "__main__":
    detect_model_drift()
    detect_data_drift()
