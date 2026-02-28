
import yfinance as yf
import pandas as pd
import numpy as np

# ===== Settings =====
CAPITAL = 100000
RISK_PCT = 0.01
MAX_POS_PCT = 0.20


def normalize_yf_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0)
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    vol = df["Volume"]

    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["RSI14"] = rsi(close, 14)

    df["HH20_prev"] = close.rolling(20).max().shift(1)
    df["Breakout20"] = close > df["HH20_prev"]

    df["Vol20"] = vol.rolling(20).mean()
    df["VolConfirm"] = vol > (1.2 * df["Vol20"])

    df["ATR14"] = atr(df, 14)
    return df


def apply_rules(latest: pd.Series) -> dict:
    # Pullback mode rules
    close = latest["Close"]
    trend_ok = (close > latest["SMA50"]) and (latest["SMA50"] > latest["SMA200"])

    dist20 = (close / latest["SMA20"]) - 1
    dist50 = (close / latest["SMA50"]) - 1
    near_ma = (0 <= dist20 <= 0.035) or (0 <= dist50 <= 0.035)

    rsi_ok = (45 <= latest["RSI14"] <= 70)

    passed = trend_ok and near_ma and rsi_ok

    score = 0
    score += 4 if trend_ok else 0
    score += 3 if near_ma else 0
    score += 2 if rsi_ok else 0
    score += 1 if bool(latest["VolConfirm"]) else 0

    reason = []
    if trend_ok: reason.append("Trend OK")
    if near_ma: reason.append("Near MA20/50")
    if rsi_ok: reason.append("RSI OK")
    if bool(latest["VolConfirm"]): reason.append("Vol bonus")
    if not passed: reason.append("Failed rule(s)")

    return {"pass": passed, "score": score, "reason": "; ".join(reason)}


def scan_universe(tickers, period="1y") -> pd.DataFrame:
    rows = []

    for t in tickers:
        try:
            df = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False)
            df = normalize_yf_df(df, t)

            if df.empty or len(df) < 220:
                rows.append({"Ticker": t, "Pass": False, "Score": 0, "Reason": "Not enough rows"})
                continue

            df = compute_features(df).dropna()
            if df.empty:
                rows.append({"Ticker": t, "Pass": False, "Score": 0, "Reason": "Indicators NaN"})
                continue

            latest = df.iloc[-1]
            res = apply_rules(latest)

            entry = float(latest["Close"])
            atr14 = float(latest["ATR14"])

            stop = entry - (2 * atr14)
            target = entry + (3 * atr14)

            risk_per_share = entry - stop
            reward_per_share = target - entry
            rr = (reward_per_share / risk_per_share) if risk_per_share > 0 else np.nan

            risk_amount = CAPITAL * RISK_PCT
            qty = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            pos_value = qty * entry

            # Position cap
            max_pos_value = CAPITAL * MAX_POS_PCT
            if pos_value > max_pos_value:
                qty = int(max_pos_value / entry)
                pos_value = qty * entry

            dist_sma50_pct = ((entry / float(latest["SMA50"])) - 1) * 100

            rows.append({
                "Ticker": t,
                "Pass": res["pass"],
                "Score": res["score"],
                "Entry": entry,
                "Stop": stop,
                "Target": target,
                "RR": rr,
                "Qty": qty,
                "PosValue": pos_value,
                "ATR14": atr14,
                "RSI14": float(latest["RSI14"]),
                "Dist_SMA50_%": dist_sma50_pct,
                "VolConfirm": bool(latest["VolConfirm"]),
                "Reason": res["reason"],
            })

        except Exception as e:
            rows.append({"Ticker": t, "Pass": False, "Score": 0, "Reason": f"Error: {e}"})

    return pd.DataFrame(rows).sort_values(["Pass", "Score"], ascending=[False, False]).reset_index(drop=True)


if __name__ == "__main__":
    from tickers import TICKERS

    results = scan_universe(TICKERS, period="1y")
    top10 = results[results["Pass"] == True].head(10)
    top10.to_csv("top10_swing_today.csv", index=False)
    print(top10)
