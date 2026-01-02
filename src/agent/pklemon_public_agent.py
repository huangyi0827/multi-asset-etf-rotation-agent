from dataclasses import dataclass
import numpy as np
import pandas as pd

def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    return (s - s.mean()) / std if std != 0 else s * 0.0

def softmax(scores: pd.Series, temp: float = 1.0) -> pd.Series:
    x = scores / max(temp, 1e-9)
    x = x - x.max()
    ex = np.exp(x)
    return ex / ex.sum()

@dataclass
class PKLemonPublicConfig:
    start: str
    end: str
    rebalance_freq: int = 5
    lookback_mom: int = 20
    lookback_vol: int = 60
    topk_per_class: int = 3
    softmax_temp: float = 0.8
    min_weight: float = 0.01

    class_weights: dict = None  # e.g. {"EQUITY_CORE":0.45,...}

class PKLemonPublicAgent:
    def __init__(self, cfg: PKLemonPublicConfig):
        if cfg.class_weights is None:
            cfg.class_weights = {
                "EQUITY_CORE": 0.45,
                "EQUITY_SAT": 0.20,
                "BOND": 0.25,
                "COMMODITY": 0.10
            }
        self.cfg = cfg

    def _pivot_close(self, prices: pd.DataFrame) -> pd.DataFrame:
        px = prices.pivot(index="date", columns="code", values="close").sort_index()
        return px.ffill()

    def _score(self, px_close: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        # 去敏版信号：动量 - 0.5*波动（结构像真的，内容不敏感）
        mom = px_close.pct_change(self.cfg.lookback_mom).loc[asof]
        vol = px_close.pct_change().rolling(self.cfg.lookback_vol).std().loc[asof]
        common = mom.dropna().index.intersection(vol.dropna().index)
        mom, vol = mom.loc[common], vol.loc[common]
        return zscore(mom) - 0.5 * zscore(vol)

    def generate_weights(self, universe: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        start = pd.to_datetime(self.cfg.start)
        end = pd.to_datetime(self.cfg.end)

        px_close = self._pivot_close(prices)
        dates = px_close.index[(px_close.index >= start) & (px_close.index <= end)]

        # rebalance dates
        rebalance_dates = dates[:: self.cfg.rebalance_freq]
        cls_map = dict(zip(universe["code"], universe["asset_class"]))

        out_rows = []
        for dt in rebalance_dates:
            score = self._score(px_close, dt)

            w = pd.Series(0.0, index=px_close.columns)

            for cls, base_w in self.cfg.class_weights.items():
                cls_codes = [c for c in score.index if cls_map.get(c) == cls]
                if not cls_codes or base_w <= 0:
                    continue
                cls_scores = score.loc[cls_codes].sort_values(ascending=False)
                selected = cls_scores.head(min(self.cfg.topk_per_class, len(cls_scores)))

                w_cls = softmax(selected, temp=self.cfg.softmax_temp) * base_w
                w.loc[w_cls.index] = w_cls

            # min weight cleanup + renorm
            w[w.abs() < self.cfg.min_weight] = 0.0
            if w.sum() > 0:
                w = w / w.sum()

            for code, weight in w[w > 0].items():
                out_rows.append({"date": dt, "code": code, "weight": float(weight)})

        weights_df = pd.DataFrame(out_rows)
        return weights_df
