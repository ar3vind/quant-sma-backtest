# Momentum SMA Crossover Backtest (Python)

Backtests a simple **20/100-day SMA crossover** on a single ticker (default: AAPL).  
Outputs an equity curve image and a text summary with Sharpe & Max Drawdown.

## How to run
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate
pip install -r requirements.txt
python backtest.py
