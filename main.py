#!/usr/bin/env python3
"""
main.py
- FastAPI wrapper around bot.py logic
- Exposes endpoints for pools, predictions, activity, strategies, and health
"""

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import numpy as np

# Import helpers from bot.py
from bot import (
    load_model,
    fetch_live_data,
    align_features,
    predict_apy,
    estimate_profit,
    execute_strategy,
    PROFIT_THRESHOLD,
    CONF_THRESHOLD,
    POOL_ID,
)


app = FastAPI(title="DeFi ML Bot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)


model = load_model()
model_features = model.get_booster().feature_names



def safe_cast(val):
    """Convert numpy/pandas types into native Python types for JSON"""
    if isinstance(val, (np.generic,)):
        return val.item()
    return val


def dataframe_to_json(df: pd.DataFrame):
    """Convert DataFrame safely to JSON serializable dict"""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df.where(pd.notnull(df), None)
    return df.applymap(safe_cast).to_dict(orient="records")


def get_prediction_snapshot():
    """Fetch, predict, and calculate profit + risk in one step"""
    snapshot = fetch_live_data()
    if snapshot.empty:
        return {"error": "No data available"}

    X_row = snapshot.drop(columns=["timestamp", "apy", "tvlUsd"], errors="ignore")
    X_row = align_features(X_row, model_features)

    pred_apy, ci, conf = predict_apy(model, X_row)
    pred_apy = safe_cast(pred_apy)
    ci = [safe_cast(ci[0]), safe_cast(ci[1])] if ci else [0.0, 0.0]
    conf = safe_cast(conf)

    tvl = safe_cast(snapshot["tvlUsd"].iloc[0])
    profit = safe_cast(estimate_profit(pred_apy, pred_apy + 2, tvl))
    risk_score = safe_cast((ci[1] - ci[0]) / pred_apy) if pred_apy else 0.0

    return {
        "pool_id": POOL_ID,
        "predicted_apy": pred_apy,
        "confidence_interval": ci,
        "confidence": conf,
        "tvl": tvl,
        "estimated_profit": profit,
        "risk_score": risk_score,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }



@app.get("/")
def root():
    return {"message": "DeFi ML Bot API is running "}


@app.get("/pools")
def pools():
    """Return live pool snapshot with APY + TVL"""
    snap = fetch_live_data()
    return JSONResponse(content=dataframe_to_json(snap))


@app.get("/pools/summary")
def pools_summary():
    """Return summarized pool stats (avg, max, min APY, total TVL)"""
    snap = fetch_live_data()
    if snap.empty:
        return {"error": "No data available"}

    summary = {
        "pool_count": safe_cast(len(snap)),
        "avg_apy": safe_cast(snap["apy"].mean()),
        "max_apy": safe_cast(snap["apy"].max()),
        "min_apy": safe_cast(snap["apy"].min()),
        "total_tvl": safe_cast(snap["tvlUsd"].sum()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return summary


@app.get("/predictions")
def predictions():
    """Return model prediction + profit + risk score"""
    return get_prediction_snapshot()


@app.get("/strategies/rebalance")
def rebalance():
    """Simulate rebalance if thresholds are met and return signed signal"""
    snapshot = get_prediction_snapshot()

    if snapshot.get("error"):
        return {"status": "skipped", "reason": "No data available"}

    if snapshot["estimated_profit"] > PROFIT_THRESHOLD and snapshot["confidence"] >= CONF_THRESHOLD:
        signed_signal = execute_strategy("Aave", "Compound", snapshot["tvl"] * 0.1)
        return {
            "status": "executed",
            "signal": signed_signal,
            "metrics": snapshot
        }
    else:
        return {"status": "skipped", "reason": "No safe opportunity", "metrics": snapshot}


@app.get("/activity")
def activity():
    """Returns last activity from log file"""
    try:
        with open("bot.log", "r") as f:
            lines = f.readlines()
        return {"last_activity": lines[-5:]}  # last 5 lines
    except FileNotFoundError:
        return {"last_activity": []}


@app.get("/system/health")
def system_health():
    """Simple health check"""
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat() + "Z",
    }



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)