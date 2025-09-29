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

def add_indicators(df: pd.DataFrame, short_w: int, long_w: int) -> pd.DataFrame:
    df = df.copy()
    df["SMA_short"] = df["Close"].rolling(short_w, min_periods=short_w).mean()
    df["SMA_long"]  = df["Close"].rolling(long_w,  min_periods=long_w).mean()
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Signal"] = 0
    ok = df["SMA_short"].notna() & df["SMA_long"].notna()
    df.loc[ok, "Signal"] = (df.loc[ok, "SMA_short"] > df.loc[ok, "SMA_long"]).astype(int)
    df["Position"] = df["Signal"].diff().fillna(0)
    return df

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Close_ret"]  = df["Close"].pct_change().fillna(0.0)
    df["Signal_lag"] = df["Signal"].shift(1).fillna(0.0)  # avoid look-ahead
    df["Strat_ret"]  = df["Close_ret"] * df["Signal_lag"]
    df["BH_curve"]   = (1.0 + df["Close_ret"]).cumprod()
    df["STR_curve"]  = (1.0 + df["Strat_ret"]).cumprod()
    return df

def plot_curves(df: pd.DataFrame, plot_path: str, initial_capital: float, ticker: str):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["STR_curve"] * initial_capital, label="Strategy")
    plt.plot(df.index, df["BH_curve"]  * initial_capital, label="Buy & Hold")
    plt.legend(); plt.title(f"{ticker} Strategy vs Buy & Hold")
    plt.xlabel("Date"); plt.ylabel("Portfolio Value ($)")
    plt.tight_layout(); plt.savefig(plot_path, bbox_inches="tight"); plt.close()

