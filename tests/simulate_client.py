import requests
import time
import random

URL = "http://localhost:8000/predict"

def generate_input(drift=False):
    """
    Generates dummy input data.
    If drift=True, shifts the distribution of feature2 to simulate drift.
    """
    feature1 = random.gauss(1.0, 0.2)  # stays consistent
    if drift:
        feature2 = random.gauss(3.0, 0.5)  # shifted distribution
    else:
        feature2 = random.gauss(1.0, 0.2)  # original distribution

    return {"feature1": feature1, "feature2": feature2}

def simulate_requests(num_requests=200, drift_after=100, delay=0.1):
    for i in range(num_requests):
        drift = i >= drift_after
        payload = generate_input(drift=drift)
        try:
            r = requests.post(URL, json=payload)
            print(f"[{i}] Sent: {payload} | Status: {r.status_code} | Prediction: {r.json()['prediction']}")
        except Exception as e:
            print(f"[{i}] Failed to send request: {e}")
        
        time.sleep(delay)

if __name__ == "__main__":
    simulate_requests()