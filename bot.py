#!/usr/bin/env python3
"""
bot.py
- Uses trained model.pkl
- Monitors live pool data every 5 minutes
- Estimates APY/profit
- Signs rebalance signals using EIP712Signer
- Logs activity to file and prints pretty tables with colored risk levels
- Dry-run execution
"""

import os
import time
import joblib
import pandas as pd
import logging
from datetime import datetime
from dotenv import load_dotenv
from tabulate import tabulate
from eth_account import Account
from eth_account.messages import encode_defunct
import hashlib
import json

from src.data.fetcher import DefiLlamaFetcher
from src.data.feature_engineer import FeatureEngineer


load_dotenv()
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
CHAIN_ID = int(os.getenv("CHAIN_ID", 1))
BOT_ADDRESS = os.getenv("ML_SIGNER_ADDRESS")

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.6))
PROFIT_THRESHOLD = float(os.getenv("PROFIT_THRESHOLD", 1.0))
SLEEP_SECONDS = 300  # 5 minutes

MODEL_PATH = "models/model.pkl"
POOL_ID = "aa70268e-4b52-42bf-a116-608b370f9501"


logging.basicConfig(
    filename="bot.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler()) 


class EIP712Signer:
    def __init__(self, private_key: str, chain_id: int):
        self.account = Account.from_key(private_key)
        self.chain_id = chain_id

    def sign_rebalance_signal(self, signal: dict) -> dict:
        message_data = {
            "fromStrategy": signal["fromStrategy"],
            "toStrategy": signal["toStrategy"],
            "amount": signal["amount"],
            "timestamp": signal["timestamp"]
        }
        # Create message hash
        message_str = json.dumps(message_data, sort_keys=True)
        message_hash = hashlib.sha256(message_str.encode()).hexdigest()

        # Sign the hash with eth_account
        msg = encode_defunct(hexstr=f"0x{message_hash}")
        signed = self.account.sign_message(msg)

        signal["signature"] = signed.signature.hex()
        signal["messageHash"] = f"0x{message_hash}"
        signal["signer"] = self.account.address
        return signal

signer = EIP712Signer(PRIVATE_KEY, CHAIN_ID)


def load_model():
    return joblib.load(MODEL_PATH)

def fetch_live_data():
    fetcher = DefiLlamaFetcher()
    df = fetcher.get_pool_chart(POOL_ID)
    fe = FeatureEngineer()
    df = fe.create_features(df)
    return df.tail(1)

def predict_apy(model, X_row):
    pred = model.predict(X_row)[0]
    ci_low, ci_high = pred * 0.9, pred * 1.1
    confidence = 0.8 if (ci_high - ci_low) < abs(pred) * 0.3 else 0.5
    return pred, (ci_low, ci_high), confidence

def estimate_profit(pred_a, pred_b, tvl, gas_cost=5, slippage=0.5):
    yield_gain = (pred_b - pred_a) * tvl / 100
    net_profit = yield_gain - (gas_cost + slippage)
    return net_profit

def execute_strategy(from_protocol, to_protocol, amount) -> dict:
    """Simulate rebalance, sign, log, and return structured signal"""
    signal = {
        "fromStrategy": from_protocol,
        "toStrategy": to_protocol,
        "amount": float(amount),
        "timestamp": int(time.time())
    }
    signed_signal = signer.sign_rebalance_signal(signal)

    logging.info(f"⚡ Simulated move {amount:.2f} from {from_protocol} → {to_protocol}")
    logging.info(f" Signed rebalance signal: {signed_signal}")

    return signed_signal

def align_features(X_row: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """Ensure X_row has the same features as the model expects."""
    for col in model_features:
        if col not in X_row.columns:
            X_row[col] = 0
    X_row = X_row[model_features]  # reorder + drop extras
    return X_row


def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

def risk_color(risk_score):
    """Green = low, Yellow = medium, Red = high"""
    if risk_score < 0.1:
        return "32"  # green
    elif risk_score < 0.3:
        return "33"  # yellow
    else:
        return "31"  # red


def pretty_print(snapshot, pred_apy, ci, conf, profit):
    risk_score = (ci[1] - ci[0]) / pred_apy if pred_apy != 0 else 0
    colored_risk = color_text(f"{risk_score:.2f}", risk_color(risk_score))
    table = [
        ["Pool", POOL_ID],
        ["Predicted APY", f"{pred_apy:.2f}%"],
        ["Confidence", f"{conf:.2f}"],
        ["Confidence Interval", f"{ci[0]:.2f} – {ci[1]:.2f}"],
        ["TVL", f"${float(snapshot['tvlUsd'].iloc[0]):,.2f}"],
        ["Estimated Profit", f"${profit:,.2f}"],
        ["Risk Score", colored_risk]
    ]
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))
    logging.info(f"Predicted APY: {pred_apy:.2f}%, Profit: ${profit:,.2f}, Risk Score: {risk_score:.2f}")


def run_bot():
    logging.info(f" Bot started at {datetime.now()}, address {BOT_ADDRESS}")
    model = load_model()
    model_features = model.get_booster().feature_names

    while True:  # infinite monitoring loop
        snapshot = fetch_live_data()
        X_row = snapshot.drop(columns=["timestamp", "apy", "tvlUsd"], errors="ignore")
        X_row = align_features(X_row, model_features)

        pred_apy, ci, conf = predict_apy(model, X_row)
        pred_a, pred_b = pred_apy, pred_apy + 2
        tvl = float(snapshot["tvlUsd"].iloc[0])
        profit = estimate_profit(pred_a, pred_b, tvl)

        pretty_print(snapshot, pred_apy, ci, conf, profit)

        if profit > PROFIT_THRESHOLD and conf >= CONF_THRESHOLD:
            signed = execute_strategy("Aave", "Compound", tvl * 0.1)
            logging.info(f" Executed strategy: {signed}")
        else:
            logging.info(" No safe opportunity.")

        logging.info(f"💤 Sleeping for {SLEEP_SECONDS / 60:.0f} minutes...\n")
        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    run_bot()
