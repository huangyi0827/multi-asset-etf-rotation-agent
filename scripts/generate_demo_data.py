import numpy as np
import pandas as pd

def make_ohlc(symbols, start="2022-01-01", end="2025-12-26", seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)

    def params(code):
        # 简单模拟：债低波、商品中波、权益高波
        if code.startswith("511"):
            return 0.00006, 0.002
        if code in ["518880"]:
            return 0.00010, 0.008
        return 0.00018, 0.012

    rows = []
    for code in symbols:
        mu, sigma = params(code)
        rets = rng.normal(mu, sigma, size=len(dates))
        close = 1.0 * np.exp(np.cumsum(rets))

        # 生成OHLC（保证 high>=max(open,close), low<=min(open,close)）
        open_ = close * (1 + rng.normal(0, sigma/5, size=len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, sigma/6, size=len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, sigma/6, size=len(dates))))

        df = pd.DataFrame({
            "date": dates,
            "code": code,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_factor": 1.0
        })
        rows.append(df)

    return pd.concat(rows, ignore_index=True)

if __name__ == "__main__":
    uni = pd.read_csv("demo_data/universe.csv", dtype={"code": str})
    symbols = uni["code"].astype(str).tolist()
    price = make_ohlc(symbols)
    price.to_csv("demo_data/prices.csv", index=False)
    print("Wrote demo_data/prices.csv with columns:", list(price.columns))
