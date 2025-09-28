import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    sys.stderr.write("Install deps: pip install -r requirements.txt\n")
    raise

TICKER = "AAPL"
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"
SHORT_WINDOW = 20
LONG_WINDOW  = 100
INITIAL_CAPITAL = 10_000
PLOT_PATH   = "equity_curve.png"
REPORT_PATH = "backtest_summary.txt"

def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} between {start} and {end}")
    out = df[["Close"]].copy().dropna()
    return out
