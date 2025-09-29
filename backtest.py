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

def write_report(report_path: str, ticker: str, start: str, end: str, metrics: dict, plot_path: str):
    text = f"""
Momentum SMA Crossover Backtest

Ticker: {ticker}
Period: {start} to {end}

Total Return (Strategy): {metrics['total_return_pct']:.2f}%
Sharpe (annualized): {metrics['sharpe']:.2f}
Max Drawdown: {metrics['max_drawdown_pct']:.2f}%

See plot: {plot_path}
""".strip() + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)


def performance_metrics(df: pd.DataFrame) -> dict:
    eps = 1e-12
    total_return = (df["STR_curve"].iloc[-1] - 1.0) * 100.0
    vol = df["Strat_ret"].std()
    sharpe = 0.0 if (vol is None or vol < eps) else (df["Strat_ret"].mean() / vol) * math.sqrt(252)
    max_dd = (df["STR_curve"] / df["STR_curve"].cummax() - 1.0).min() * 100.0
    return {
        "total_return_pct": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(max_dd),
    }

def main():
    df = download_prices(TICKER, START_DATE, END_DATE)
    df = add_indicators(df, SHORT_WINDOW, LONG_WINDOW)
    df = generate_signals(df)
    df = compute_returns(df)
    metrics = performance_metrics(df)
    plot_curves(df, PLOT_PATH, INITIAL_CAPITAL, TICKER)
    write_report(REPORT_PATH, TICKER, START_DATE, END_DATE, metrics, PLOT_PATH)
    print(f"Saved: {PLOT_PATH}, {REPORT_PATH}")
    print(f"Return: {metrics['total_return_pct']:.2f}%  Sharpe: {metrics['sharpe']:.2f}  MaxDD: {metrics['max_drawdown_pct']:.2f}%")

if __name__ == "__main__":
    main()
