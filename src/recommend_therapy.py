"""
Therapy recommendation helper.
This is a decision-support demo, not a clinical recommendation.
"""
import joblib
import pandas as pd

THERAPIES = {
    "Speech_Therapy_Need": "Speech Therapy",
    "Occupational_Therapy_Need": "Occupational Therapy",
    "Behaviour_Therapy_Need": "Behaviour Therapy",
    "Parent_Training_Need": "Parent Training",
    "Social_Skills_Support_Need": "Social Skills Support",
}

def recommend(model_path, profile):
    model = joblib.load(model_path)
    X = pd.DataFrame([profile])
    pred = model.predict(X)[0]
    return [THERAPIES[k] for k, v in zip(THERAPIES, pred) if int(v) == 1]
