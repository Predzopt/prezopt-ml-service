# prezopt-ml-service

Machine learning service for Prezopt Protocol — predicts profitable yield rebalances and signs signals for on-chain execution.

## Features

- Fetches historical APY/TVL/emissions from DefiLlama, CoinGecko, Arbiscan
- Trains ensemble model (XGBoost + LSTM) to predict net yield differentials
- Serves signed EIP-712 rebalance signals via REST API
- Dockerized for deployment to AWS ECS / GCP Cloud Run

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env
python train.py  # train initial model
uvicorn src.main:app --reload  # start API