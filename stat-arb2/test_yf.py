import yfinance as yf
import time
from concurrent.futures import ThreadPoolExecutor

tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
ticks = yf.Tickers(" ".join(tickers))
start = time.time()
res = {t: ticks.tickers[t].fast_info.get('marketCap', 0) for t in tickers}
print("Time:", time.time()-start)
print(res)
