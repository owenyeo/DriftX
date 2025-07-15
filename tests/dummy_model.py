from flask import Flask, request, jsonify
import time
import random

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    time.sleep(random.uniform(0.01, 0.1))  # simulate processing delay
    return jsonify({"prediction": round(random.uniform(0, 1), 3)})

app.run(port=9000)