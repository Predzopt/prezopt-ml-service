import requests
import pandas as pd
from datetime import datetime
import time


class DefiLlamaFetcher:
    def __init__(self):
        self.base_url = "https://yields.llama.fi"

    def get_pool_chart(self, pool_id: str) -> pd.DataFrame:
        """Fetch historical APY/TVL for a pool"""
        url = f"{self.base_url}/chart/{pool_id}"
        response = requests.get(url)

        if response.status_code != 200:
            raise ConnectionError(f"DefiLlama API error {response.status_code}: {response.text}")

        data = response.json()

        # Check status field
        if data.get('status') and data['status'] != 'success':
            raise ValueError(f"DefiLlama API returned error: {data}")

        # Must have data field
        if not data.get('data'):
            raise ValueError(f"No data returned for pool {pool_id}. Full response: {data}")

        df = pd.DataFrame(data['data'])

        # Detect timestamp column
        timestamp_col = None
        for col in ['date', 'timestamp']:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            raise KeyError(f"No timestamp column found. Available columns: {df.columns.tolist()}")

        df['timestamp'] = pd.to_datetime(df[timestamp_col], unit='s', errors='coerce').dt.tz_localize(None)
        return df

    def get_pools(self) -> list:
        """Get list of all pools"""
        url = f"{self.base_url}/pools"
        response = requests.get(url)

        if response.status_code != 200:
            raise ConnectionError(f"DefiLlama API error {response.status_code}: {response.text}")

        data = response.json()
        return data.get('data', [])


class CoinGeckoFetcher:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"

    def _safe_request(self, url, params=None, max_retries=3):
        """Safe request with retry on rate-limits"""
        for attempt in range(max_retries):
            response = requests.get(url, params=params)
            
            # Handle rate-limiting
            if response.status_code == 429:
                wait = 2 ** attempt
                print(f"Rate limited by CoinGecko. Retrying in {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                raise ConnectionError(f"CoinGecko API error {response.status_code}: {response.text}")

            return response.json()

        raise TimeoutError(f"CoinGecko API failed after {max_retries} retries")

    def get_token_prices(self, token_id: str, days: str = "365") -> pd.DataFrame:
        """Fetch historical token prices"""
        url = f"{self.base_url}/coins/{token_id}/market_chart"
        params = {"vs_currency": "usd", "days": days}
        data = self._safe_request(url, params=params)

        # Validate data
        if 'error' in data:
            raise ValueError(f"CoinGecko API error: {data['error']}")

        if 'prices' not in data:
            raise KeyError(f"'prices' not found in response. Keys: {list(data.keys())}. Full response: {data}")

        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize(None)
        return df


# ✅ Example usage
if __name__ == "__main__":
    cg = CoinGeckoFetcher()
    df = cg.get_token_prices("ethereum", days="30")
    print(df.head())

    llama = DefiLlamaFetcher()
    pools = llama.get_pools()
    print(f"Found {len(pools)} pools")
