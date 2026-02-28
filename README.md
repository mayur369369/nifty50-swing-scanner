# nifty50-swing-scanner
Systematic Nifty50 swing trading scanner using ATR risk management and pullback strategy

# Nifty50 Swing Scanner

This project is a systematic rule-based swing trading scanner for Nifty 50 stocks.

## Strategy Logic

The system applies:

- Trend filter: Close > SMA50 > SMA200
- Pullback logic: Price near SMA20 or SMA50
- RSI filter: RSI between 45–70
- ATR-based stop loss (2 × ATR14)
- Target: 3 × ATR14
- Risk per trade: 1% of capital
- Max position size: 20% of capital

## Output

The scanner generates:

- Entry price
- Stop loss
- Target price
- Risk:Reward ratio
- Quantity to buy
- Position value

## Files

- `Nifty50_Swing_Scanner.ipynb` – Main notebook
- `top10_swing_today.csv` – Latest top 10 swing picks
- `nifty50_swing_scan_YYYY-MM-DD.csv` – Full scan results

## Future Improvements

- Daily automated GitHub Action run
- Email alert with Top 10 picks
- Backtesting engine
- Market trend filter

---

Built using Python, pandas, yfinance, NumPy.
