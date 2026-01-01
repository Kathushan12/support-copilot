import os
import joblib
from src.config import MODELS_DIR

def load_model():
    return joblib.load(os.path.join(MODELS_DIR, "triage_model.joblib"))

def predict_category(text: str):
    model = load_model()
    pred = model.predict([text])[0]
    proba = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba([text])[0]
        proba = float(max(p))
    return pred, proba
