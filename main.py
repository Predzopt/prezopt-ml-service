#!/usr/bin/env python3
"""
main.py
- FastAPI wrapper around enhanced bot.py logic
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

# Import updated helpers from bot.py
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
)

app = FastAPI(title="DeFi ML Bot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
try:
    model = load_model()
    # Safely get feature names (XGBoost compatible)
    if hasattr(model, "get_booster"):
        model_features = model.get_booster().feature_names
    elif hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_.tolist()
    else:
        # Fallback — update this to match your training features
        model_features = ["apy_7d_ma", "tvl_change", "volumeUsd", "fee_apy", "utilization_rate"]
    logging.info(f"✅ Model loaded with {len(model_features)} features")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
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
    if df.empty:
        return []
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df.where(pd.notnull(df), None)
    return df.applymap(safe_cast).to_dict(orient="records")


def get_prediction_snapshot():
    """Fetch, predict, and calculate profit + risk in one step (single pool)"""
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
    return {"message": "DeFi ML Bot API (v2) is running with 4-strategy support"}


@app.get("/pools")
def pools():
    """Return live pool snapshot with APY + TVL"""
    try:
        snap = fetch_live_data()
        return JSONResponse(content=dataframe_to_json(snap))
    except Exception as e:
        return {"error": str(e)}


@app.get("/pools/summary")
def pools_summary():
    """Return summarized pool stats (avg, max, min APY, total TVL)"""
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
    """Return per-strategy APY predictions"""
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
    """Simulates: contract.getStrategyAllocations()"""
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
    """Dynamically simulate rebalance(from, to) among 4 strategies"""
    if model is None:
        return {"status": "error", "reason": "Model not loaded"}

    try:
        # Get current state
        allocations, current_apys, _ = get_strategy_allocations()
        snapshot = fetch_live_data()
        if snapshot.empty:
            return {"status": "skipped", "reason": "No live data"}

        # Predict APY for all 4 strategies
        predicted_apys = predict_all_apys(model, model_features, snapshot)
        rebalance_plan = find_best_rebalance(allocations, predicted_apys)

        if not rebalance_plan:
            return {
                "status": "skipped",
                "reason": "Already optimally allocated",
                "current_allocations": dict(zip(STRATEGY_NAMES, allocations)),
                "predicted_apys": dict(zip(STRATEGY_NAMES, predicted_apys))
            }

        from_idx, to_idx, amount = rebalance_plan
        from_name = STRATEGY_NAMES[from_idx]
        to_name = STRATEGY_NAMES[to_idx]

        # Estimate profit using current APY (from) and predicted APY (to)
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
        logging.error(f"Rebalance error: {e}", exc_info=True)
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
        "supported_strategies": STRATEGY_NAMES,
        "contract_address": "0x709900553fE09E934243282F764A806A50Acfc21",
        "time": datetime.utcnow().isoformat() + "Z"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)