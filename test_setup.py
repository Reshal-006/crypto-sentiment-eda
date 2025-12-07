import pandas as pd
import yfinance as yf
import requests

print("Pandas version:", pd.__version__)
print("YFinance version:", yf.__version__)

url = "https://api.alternative.me/fng/?limit=1&format=json"
r = requests.get(url, timeout=10)
print("Fear & Greed API status:", r.status_code)
print("Sample data:", r.json()["data"][0])