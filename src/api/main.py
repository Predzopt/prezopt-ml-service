#!/usr/bin/env python3
"""
main.py
- FastAPI wrapper around the enhanced bot.py logic
- Supports 4 strategies: Aave, Compound, Yarn, Curve
- Exposes endpoints for pools, predictions, strategies, rebalance, activity, and health
"""

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import numpy as np
import logging

# Import from bot.py (now enhanced with 4-strategy logic)
from bot import (
    load_model,
    fetch_live_data,
    align_features,
    predict_apy,
    estimate_profit,
    STRATEGY_NAMES,
    get_strategy_allocations,
    predict_all_apys,
    find_best_rebalance,
    simulate_rebalance,
    PROFIT_THRESHOLD,
    CONF_THRESHOLD,
    POOL_ID,
    MODEL_PATH,
)

app = FastAPI(title="DeFi ML Bot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
try:
    model = load_model()
    model_features = model.get_booster().feature_names
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None
    model_features = []


def safe_cast(val):
    """Convert numpy/pandas types to native Python types for JSON"""
    if isinstance(val, (np.generic,)):
        return val.item()
    if pd.isna(val):
        return None
    return val


def dataframe_to_json(df: pd.DataFrame):
    """Safely convert DataFrame to JSON-serializable dict"""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df.where(pd.notnull(df), None)
    return df.applymap(safe_cast).to_dict(orient="records")


@app.get("/")
def root():
    return {"message": "DeFi ML Bot API (v2) is running with 4-strategy support"}


@app.get("/pools")
def pools():
    """Return live pool snapshot"""
    try:
        snap = fetch_live_data()
        return JSONResponse(content=dataframe_to_json(snap))
    except Exception as e:
        return {"error": str(e)}


@app.get("/pools/summary")
def pools_summary():
    """Return summarized pool stats"""
    try:
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
    except Exception as e:
        return {"error": str(e)}


@app.get("/predictions")
def predictions():
    """Return per-strategy APY predictions based on ML model"""
    if model is None:
        return {"error": "Model not loaded"}

    try:
        snapshot = fetch_live_data()
        if snapshot.empty:
            return {"error": "No live data"}

        predicted_apys = predict_all_apys(model, model_features, snapshot)
        current_allocations, current_apys, _ = get_strategy_allocations()

        strategies = []
        for i, name in enumerate(STRATEGY_NAMES):
            strategies.append({
                "name": name,
                "index": i,
                "current_apy": safe_cast(current_apys[i]),
                "predicted_apy": safe_cast(predicted_apys[i]),
                "allocation_usd": safe_cast(current_allocations[i]),
                "delta_apy": safe_cast(predicted_apys[i] - current_apys[i])
            })

        return {
            "strategies": strategies,
            "base_pool_id": POOL_ID,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/strategies")
def strategies():
    """
    Simulates: contract.getStrategyAllocations()
    Returns allocations, APYs, and names for all 4 strategies.
    """
    try:
        allocations, apys, names = get_strategy_allocations()
        strategies = []
        for i in range(len(names)):
            strategies.append({
                "name": names[i],
                "index": i,
                "allocation_usd": safe_cast(allocations[i]),
                "apy": safe_cast(apys[i])
            })
        return {
            "strategies": strategies,
            "contract_address": "0x709900553fE09E934243282F764A806A50Acfc21",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/strategies/rebalance")
def rebalance():
    """
    Simulates: contract.rebalance(fromStrategy, toStrategy)
    Returns signed EIP-712 signal if profitable and confident.
    """
    if model is None:
        return {"status": "error", "reason": "Model not loaded"}

    try:
        # Get current state
        allocations, current_apys, _ = get_strategy_allocations()
        snapshot = fetch_live_data()
        if snapshot.empty:
            return {"status": "skipped", "reason": "No live data"}

        # Predict
        predicted_apys = predict_all_apys(model, model_features, snapshot)
        rebalance_plan = find_best_rebalance(allocations, predicted_apys)

        if not rebalance_plan:
            return {
                "status": "skipped",
                "reason": "Already optimally allocated",
                "current_allocations": allocations,
                "predicted_apys": predicted_apys
            }

        from_idx, to_idx, amount = rebalance_plan
        from_name = STRATEGY_NAMES[from_idx]
        to_name = STRATEGY_NAMES[to_idx]

        # Estimate profit
        profit = estimate_profit(current_apys[from_idx], predicted_apys[to_idx], amount)

        if profit > PROFIT_THRESHOLD:
            signed_signal = simulate_rebalance(from_idx, to_idx, amount)
            return {
                "status": "executed",
                "from_strategy": from_name,
                "to_strategy": to_name,
                "from_index": from_idx,
                "to_index": to_idx,
                "amount_usd": safe_cast(amount),
                "estimated_profit_usd": safe_cast(profit),
                "signal": signed_signal,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        else:
            return {
                "status": "skipped",
                "reason": "Profit below threshold",
                "estimated_profit_usd": safe_cast(profit),
                "threshold_usd": PROFIT_THRESHOLD,
                "from_strategy": from_name,
                "to_strategy": to_name
            }

    except Exception as e:
        return {"status": "error", "reason": str(e)}


@app.get("/activity")
def activity():
    """Returns last 10 lines from bot.log"""
    try:
        with open("bot.log", "r") as f:
            lines = f.readlines()
        return {"last_activity": [line.strip() for line in lines[-10:]]}
    except FileNotFoundError:
        return {"last_activity": [], "note": "bot.log not found"}


@app.get("/system/health")
def system_health():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "failed"
    return {
        "status": "ok",
        "model": model_status,
        "model_path": MODEL_PATH,
        "time": datetime.utcnow().isoformat() + "Z",
        "supported_strategies": STRATEGY_NAMES,
        "contract_address": "0x709900553fE09E934243282F764A806A50Acfc21"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)