# src/data/fetcher.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DefiLlamaFetcher:
    def __init__(self):
        pass  # No real API calls

    def get_pool_chart(self, pool_id: str, days: int = 180) -> pd.DataFrame:
        """Generate DYNAMIC mock yield data that looks real"""
        logger.info(f"Generating mock data for pool: {pool_id}")
        
        # Create realistic date range
        end = datetime.utcnow()
        dates = [end - timedelta(days=i) for i in range(days, -1, -1)]
        
        # Base APY with realistic behavior
        base_apy = 4.5
        apy = [base_apy]
        np.random.seed(hash(pool_id) % 2**32)
        
        for i in range(1, len(dates)):
            # Weekly pattern: lower on weekends
            weekend_effect = -0.3 if dates[i].weekday() >= 5 else 0
            # Random walk with momentum
            change = np.random.normal(0, 0.15)
            new_apy = apy[-1] + change + weekend_effect
            apy.append(max(0.5, new_apy))  # Min 0.5% APY
        
        # TVL correlated with APY
        base_tvl = 1_000_000_000
        tvl = []
        for a in apy:
            effect = (a - base_apy) * 20_000_000
            noise = np.random.normal(0, base_tvl * 0.03)
            tvl_val = base_tvl + effect + noise
            tvl.append(max(10_000_000, tvl_val))
        
        volume = [t * 0.05 for t in tvl]  # 5% of TVL daily
        
        df = pd.DataFrame({
            "date": [int(d.timestamp()) for d in dates],
            "apy": apy,
            "tvlUsd": tvl,
            "volumeUsd": volume
        })
        df["timestamp"] = pd.to_datetime(df["date"], unit="s")
        return df

    def get_pools(self) -> list:
        """Return mock pool list"""
        return [{
            "pool": "aa70268e-4b52-42bf-a116-608b370f9501",
            "project": "aave",
            "symbol": "WETH",
            "chain": "Ethereum",
            "tvlUsd": 1_000_000_000,
            "apy": 4.5
        }]


class CoinGeckoFetcher:
    def __init__(self):
        pass

    def get_token_prices(self, token_id: str, days: str = "365") -> pd.DataFrame:
        """Generate DYNAMIC mock price data (ETH-like volatility)"""
        logger.info(f"Generating mock {token_id} prices")
        
        days_int = 365 if days == "max" else int(days)
        end = datetime.utcnow()
        # Hourly data for realism
        dates = [end - timedelta(hours=i) for i in range(days_int * 24, -1, -1)]
        
        start_price = 3000.0
        prices = [start_price]
        np.random.seed(hash(token_id) % 2**32)
        
        for i in range(1, len(dates)):
            # High volatility with occasional jumps
            hourly_return = np.random.normal(0, 0.015)
            if np.random.random() < 0.01:  # 1% chance of big move
                hourly_return += np.random.choice([-0.1, 0.1])
            new_price = prices[-1] * (1 + hourly_return)
            prices.append(max(100, new_price))
        
        df = pd.DataFrame({
            "timestamp": [int(d.timestamp() * 1000) for d in dates],
            "price": prices
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
