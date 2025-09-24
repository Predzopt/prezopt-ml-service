import requests
import pandas as pd
from datetime import datetime, timedelta

class DefiLlamaFetcher:
    def __init__(self):
        self.base_url = "https://yields.llama.fi"

    def get_pool_chart(self, pool_id: str) -> pd.DataFrame:
      """Fetch historical APY/TVL for a pool"""
      url = f"{self.base_url}/chart/{pool_id}"
      response = requests.get(url)
      data = response.json()
    
    # Check status
      if data.get('status') != 'success':
        raise ValueError(f"API error: {data}")
    
    # Check data
      if not data.get('data'):
        raise ValueError(f"No data returned for pool {pool_id}. Check pool ID.")
    
      df = pd.DataFrame(data['data'])
    
      # Find timestamp column
      timestamp_col = None
      for col in ['date', 'timestamp']:
          if col in df.columns:
              timestamp_col = col
              break
    
      if timestamp_col is None:
          raise KeyError(f"No timestamp column found. Available columns: {df.columns.tolist()}")
     
      df['timestamp'] = pd.to_datetime(df[timestamp_col]).dt.tz_localize(None)
      return df

    def get_pools(self) -> list:
        """Get list of all pools"""
        url = f"{self.base_url}/pools"
        response = requests.get(url)
        return response.json()['data']
class CoinGeckoFetcher:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"

    def get_token_prices(self, token_id: str, days: str = "365") -> pd.DataFrame:
        """Fetch historical token prices"""
        url = f"{self.base_url}/coins/{token_id}/market_chart"
        params = {"vs_currency": "usd", "days": days}
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check for API errors
        if 'error' in data:
            raise ValueError(f"CoinGecko API error: {data['error']}")
        
        # CoinGecko returns {'prices': [[timestamp, price], ...]}
        if 'prices' not in data:
            raise KeyError(f"'prices' not found in response. Keys: {data.keys()}")
        
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize(None)
        return df