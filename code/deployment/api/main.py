
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from sklearn.datasets import load_breast_cancer

app = FastAPI(title="Assignment 1")

class PredictRequest(BaseModel):
    features: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '/models/model.pkl'))
SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '/models/scaler.pkl'))
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
data = load_breast_cancer()
FEATURE_COUNT = data.data.shape[1]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    samples = req.features
    if not isinstance(samples, list) or len(samples) == 0:
        raise HTTPException(status_code=400, detail="`features` must be a non-empty list of samples")

    arr = np.array(samples, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != FEATURE_COUNT:
        raise HTTPException(status_code=400, detail=f"Each sample must have {FEATURE_COUNT} features")

    Xs = scaler.transform(arr)
    preds = model.predict(Xs).astype(int).tolist()
    probs = model.predict_proba(Xs).tolist()

    return PredictResponse(predictions=preds, probabilities=probs)

@app.get("/")
def root():
    return {""}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
