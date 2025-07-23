import requests
import time
import random
import numpy as np

# Initial distribution (Iris-like)
mean = np.array([5.1, 3.5, 1.4, 0.2])
std = np.array([0.3, 0.3, 0.2, 0.1])

# Drift increment per iteration
drift_rate = np.array([0.01, -0.01, 0.02, 0.005])

def generate_features(mean, std):
    return np.random.normal(loc=mean, scale=std).tolist()

def send_request(features):
    payload = {"features": features}
    try:
        res = requests.post("http://localhost:8000/predict", json=payload)
        print(f"Sent: {features} Response: {res.json()}")
    except Exception as e:
        print("Request failed:", e)

if __name__ == "__main__":
    for step in range(200):  # simulate 200 requests
        features = generate_features(mean, std)
        send_request(features)

        # simulate drift after 50 requests
        if step > 50:
            mean += drift_rate  # slowly shift the distribution

        time.sleep(0.5)  # send a request every 0.5 sec
