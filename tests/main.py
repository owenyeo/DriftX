from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class InputFeatures(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(input: InputFeatures):
    if len(input.features) != 4:
        raise HTTPException(status_code=400, detail="Expected 4 features.")

    input_np = np.array(input.features).reshape(1, -1)
    prediction = model.predict(input_np)[0]
    proba = model.predict_proba(input_np).tolist()[0]

    return {
        "prediction": int(prediction),
        "probabilities": proba
    }
