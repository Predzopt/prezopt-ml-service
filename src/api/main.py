from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import time
from src.utils.signer import EIP712Signer
from src.models.xgboost_model import XGBoostModel
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Prezopt ML Service")

# Load model
model = XGBoostModel(model_path=os.getenv("MODEL_PATH"))
try:
    model.load()
except:
    print("No model found. Train first.")

# Initialize signer
signer = EIP712Signer(
    private_key=os.getenv("PRIVATE_KEY"),
    chain_id=int(os.getenv("CHAIN_ID"))
)

class RebalanceRequest(BaseModel):
    allocations: Dict[str, int]
    pools: list
    gas_price_gwei: float
    chain_id: int

@app.post("/predict-rebalance")
async def predict_rebalance(request: RebalanceRequest):
    """Generate signed rebalance signal"""
    try:
        # TODO: Implement actual prediction logic
        # For now, return mock signal if conditions are met
        
        # Mock prediction
        from_strategy = "aave"
        to_strategy = "compound"
        amount = 1000000000  # 1000 USDC
        expected_net_gain = 1250000  # 1.25 USDC
        confidence = 0.78
        timestamp = int(time.time())
        min_output = int(amount * 0.995)  # 0.5% slippage

        # Only return signal if profitable and confident
        if expected_net_gain > 1000000 and confidence > 0.6:  # > $1, >60%
            signal = {
                "fromStrategy": from_strategy,
                "toStrategy": to_strategy,
                "amount": amount,
                "minOutput": min_output,
                "expectedNetGain": expected_net_gain,
                "confidence": confidence,
                "timestamp": timestamp,
                "keeperRewardPZT": 8  # mock
            }
            
            # Sign signal
            signed_signal = signer.sign_rebalance_signal(signal)
            return signed_signal
        else:
            raise HTTPException(status_code=400, detail="No profitable rebalance found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model.model is not None}