# src/data/fetcher.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DefiLlamaFetcher:
    def __init__(self):
        self.pool_id_to_config = {
            "aa70268e-4b52-42bf-a116-608b370f9501": {
                "name": "Aave WETH",
                "base_apy": 3.5,
                "tvl_base": 1_200_000_000,
                "volatility": 0.8,
                "trend": 0.02  # slight upward drift
            }
        }

    def get_pool_chart(self, pool_id: str, days: int = 180) -> pd.DataFrame:
        """Generate DYNAMIC mock data that looks real"""
        config = self.pool_id_to_config.get(
            pool_id,
            {"name": "Mock Pool", "base_apy": 4.0, "tvl_base": 500_000_000, "volatility": 1.0, "trend": 0.0}
        )
        
        logger.info(f"Generating mock data for pool: {pool_id} ({config['name']})")
        
        # Create date range (last 180 days)
        end_date = datetime.utcnow()
        dates = [end_date - timedelta(days=i) for i in range(days, -1, -1)]
        
        # Generate realistic APY with:
        # - Base level
        # - Random walk
        # - Weekly seasonality (lower on weekends)
        # - Volatility spikes
        np.random.seed(hash(pool_id) % 2**32)  # deterministic per pool
        apy = [config["base_apy"]]
        for i in range(1, len(dates)):
            # Weekly pattern: -0.3% on weekends
            day_of_week = dates[i].weekday()
            weekend_effect = -0.3 if day_of_week >= 5 else 0
            
            # Random walk with momentum
            change = np.random.normal(0, config["volatility"] * 0.1)
            new_apy = apy[-1] + change + weekend_effect + config["trend"]
            apy.append(max(0.1, new_apy))  # APY can't go below 0.1%
        
        # Generate TVL with mean-reversion and correlation to APY
        tvl = []
        base_tvl = config["tvl_base"]
        for i, a in enumerate(apy):
            # Higher APY → higher TVL (with lag)
            apy_effect = (a - config["base_apy"]) * 50_000_000
            noise = np.random.normal(0, base_tvl * 0.05)
            tvl_val = base_tvl + apy_effect + noise
            tvl.append(max(10_000_000, tvl_val))  # min TVL
        
        # Volume: 5% of TVL daily
        volume = [t * 0.05 for t in tvl]
        
        df = pd.DataFrame({
            "date": [int(d.timestamp()) for d in dates],
            "apy": apy,
            "tvlUsd": tvl,
            "volumeUsd": volume
        })
        
        # Add timestamp column for compatibility
        df["timestamp"] = pd.to_datetime(df["date"], unit="s")
        return df

    def get_pools(self) -> list:
        """Return mock pool list"""
        return [
            {
                "pool": "aa70268e-4b52-42bf-a116-608b370f9501",
                "project": "aave",
                "symbol": "WETH",
                "chain": "Ethereum",
                "tvlUsd": 1_200_000_000,
                "apy": 3.5
            }
        ]


class CoinGeckoFetcher:
    def __init__(self):
        pass

    def get_token_prices(self, token_id: str, days: str = "365") -> pd.DataFrame:
        """Generate realistic ETH-like price data"""
        logger.info(f"Generating mock {token_id} price data for {days} days")
        
        # Parse days
        if days == "max":
            days_int = 1000
        else:
            days_int = int(days)
        
        end_date = datetime.utcnow()
        dates = [end_date - timedelta(hours=i) for i in range(days_int * 24, -1, -1)]
        
        # Start price (ETH-like)
        start_price = 3000.0
        prices = [start_price]
        
        # Simulate realistic crypto price action:
        # - High volatility
        # - Mean reversion
        # - Occasional jumps
        np.random.seed(hash(token_id) % 2**32)
        for i in range(1, len(dates)):
            # Hourly return: mean=0, std=1.5%
            hourly_return = np.random.normal(0, 0.015)
            
            # Add occasional jumps (1% chance of ±10% move)
            if np.random.random() < 0.01:
                jump = np.random.choice([-0.1, 0.1])
                hourly_return += jump
            
            new_price = prices[-1] * (1 + hourly_return)
            prices.append(max(100, new_price))  # min price $100
        
        df = pd.DataFrame({
            "timestamp": [int(d.timestamp() * 1000) for d in dates],  # ms for CoinGecko
            "price": prices
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df