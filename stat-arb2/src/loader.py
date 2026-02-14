import yfinance as yf
import pandas as pd

import requests

import os

class DataLoader:
    def __init__(self, start_date, end_date, data_dir="data"):
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        self.tickers = []
        self.raw_prices = pd.DataFrame()
        self.clean_prices = pd.DataFrame()
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _fetch_sp500_tickers(self):
        """Scrapes the current S&P 500 tickers from Wikipedia."""
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        
        from io import StringIO
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        
        # Clean tickers for yfinance (e.g., BRK.B -> BRK-B)
        self.tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        return self.tickers

    def _get_cache_path(self):
        """Generates a filename based on the date range."""
        filename = f"sp500_prices_{self.start_date}_{self.end_date}.csv"
        return os.path.join(self.data_dir, filename)

    def download_data(self):
        """
        Downloads Adjusted Close prices from yfinance or loads from cache.
        """
        cache_path = self._get_cache_path()
        
        if os.path.exists(cache_path):
            print(f"Loading data from cache: {cache_path}")
            self.raw_prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            # If loaded from CSV, tickers might need to be populated from columns if needed elsewhere
            self.tickers = self.raw_prices.columns.tolist()
            return self.raw_prices

        if not self.tickers:
            self._fetch_sp500_tickers()
            
        print(f"Downloading data for {len(self.tickers)} tickers from {self.start_date} to {self.end_date}...")
        
        # DEBUG: limit tickers removed
        # self.tickers = self.tickers[:5]
        
        try:
            data = yf.download(
                self.tickers, 
                start=self.start_date, 
                end=self.end_date, 
                auto_adjust=False,
                threads=True,
                progress=False
            )
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise e
        
        # Isolate the 'Adj Close' column
        self.raw_prices = data['Adj Close']
        
        # Save to cache
        print(f"Saving data to cache: {cache_path}")
        self.raw_prices.to_csv(cache_path)
        
        return self.raw_prices

    def check_and_clean_missing_data(self, missing_threshold=0.05):
        """
        Audits the dataframe for NaNs, drops highly sparse columns, 
        and forward-fills the rest.
        """
        df = self.raw_prices.copy()
        
        # 1. Audit: Total Missing Data
        total_missing = df.isna().sum().sum()
        total_data_points = df.shape[0] * df.shape[1]
        print(f"--- Data Audit ---")
        print(f"Initial missing data points: {total_missing} out of {total_data_points}")
        
        # 2. Filter: Drop stocks missing more than the threshold (e.g., 5%)
        # This usually catches companies that IPO'd or entered the index recently.
        limit = len(df) * (1 - missing_threshold)
        df_cleaned = df.dropna(thresh=int(limit), axis=1)
        
        dropped_tickers = set(df.columns) - set(df_cleaned.columns)

        # Calculate missing percentage per ticker
        missing_per_ticker = df.isna().mean()
        tickers_with_missing = missing_per_ticker[missing_per_ticker > 0].sort_values(ascending=False)
        
        if not tickers_with_missing.empty:
            print("\nTickers with missing data (>0%):")
            for ticker, pct in tickers_with_missing.items():
                status = "[DROPPED]" if ticker in dropped_tickers else "[KEPT]"
                print(f"{ticker}: {pct:.2%} {status}")
        else:
            print("\nNo tickers have missing data.")
            
        print("-" * 20)
        
        print(f"Dropped {len(dropped_tickers)} tickers exceeding {missing_threshold*100}% missing data threshold.")
        if dropped_tickers:
            print(f"Dropped tickers: {sorted(list(dropped_tickers))}")
        
        # 3. Impute: Forward-fill remaining missing data (trading halts, single-day gaps)
        df_cleaned = df_cleaned.ffill()
        
        # 4. Clean: Drop any remaining rows with NaNs (usually the very first row)
        df_cleaned = df_cleaned.dropna(axis=0)
        
        self.clean_prices = df_cleaned
        print(f"Final clean price matrix shape: {self.clean_prices.shape[0]} days x {self.clean_prices.shape[1]} stocks.")
        print(f"------------------\n")
        
        return self.clean_prices