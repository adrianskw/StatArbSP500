from src.loader import DataLoader

loader = DataLoader(start_date="2022-01-01", end_date="2025-12-31")

# 1. Fetch raw prices of S&P500 (we are ignoring dividends)
loader.download_data()

# 2. Run the missing data audit and clean
# NOTE: This is df
clean_price_matrix = loader.check_and_clean_missing_data(missing_threshold=0.05)


