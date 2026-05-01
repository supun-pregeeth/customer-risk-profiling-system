import os
import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_risk_model.pkl")

model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "txCount30d",
    "totalAmount30d",
    "avgTxnAmount30d",
    "maxTxnAmount30d",
    "amountStddev30d",
    "spendVelocity7dVsPrev7d",
    "txVelocity7dVsPrev7d",
    "ecomRatio30d",
    "posRatio30d",
    "atmRatio30d",
    "uniqueMerchantCount30d",
    "riskyMerchantCount30d",
    "newDeviceFlagRecent",
    "newCountryFlagRecent"
]

@app.get("/")
def home():
    return {"message": "ML service running"}

@app.post("/predict")
def predict(data: dict):
    try:
        print("=== RECEIVED DATA ===")
        print(data)

        df = pd.DataFrame([data])

        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            raise Exception(f"Missing columns: {missing}")

        df = df[FEATURE_COLUMNS]

        df["newDeviceFlagRecent"] = df["newDeviceFlagRecent"].astype(int)
        df["newCountryFlagRecent"] = df["newCountryFlagRecent"].astype(int)

        df = df.fillna(0)

        print("=== FINAL DATAFRAME SENT TO MODEL ===")
        print(df)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df)[0][1]
            risk_score = float(prob) * 100
        else:
            pred = model.predict(df)[0]
            risk_score = float(pred) * 100

        if risk_score < 40:
            risk_level = "Low"
        elif risk_score < 70:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "error": None
        }

    except Exception as e:
        print("ML ERROR:", str(e))
        print("RECEIVED COLUMNS:", list(data.keys()))

        return {
            "risk_score": None,
            "risk_level": None,
            "error": str(e),
            "received_columns": list(data.keys())
        }