import pandas as pd

class FeatureEngineer:
    def __init__(self):
        pass

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML model"""
        df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['apy', 'tvlUsd']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found. Available: {df.columns.tolist()}")
        
        # Rolling volatility (smaller windows)
        df['apy_vol'] = df['apy'].rolling(window=3).std()
        
        # TVL growth rate
        df['tvl_growth'] = df['tvlUsd'].pct_change(periods=1)
        
        # Lagged features (fewer lags)
        df['apy_lag_1'] = df['apy'].shift(1)
        df['tvl_lag_1'] = df['tvlUsd'].shift(1)
        
        return df