from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd

from src.router.public_rule_router import PublicRuleRouter, RouterConfig
from src.router.audit import AuditLogger
from src.data.calendar import month_end_rebalance_days
from src.portfolio.sticky_topn import sticky_topn
from src.portfolio.shrink import turnover_pred_l1, adaptive_shrink_lambda

def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    return (s - s.mean()) / std if std != 0 else s * 0.0

def softmax(scores: pd.Series, temp: float = 1.0) -> pd.Series:
    x = scores / max(temp, 1e-9)
    x = x - x.max()
    ex = np.exp(x)
    return ex / ex.sum()

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))

def _renorm(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(float(v) for v in w.values())
    if s <= 1e-12:
        return {}
    return {k: float(v) / s for k, v in w.items() if float(v) > 0}

@dataclass
class PKLemonPublicConfig:
    start: str
    end: str

    # rebalance
    rebalance_mode: str = "month_end"  # month_end | fixed
    rebalance_freq: int = 5            # only if fixed
    include_first_rebalance: bool = True

    # signals (public surrogate)
    lookback_mom: int = 20
    lookback_vol: int = 60

    # within-class
    within_temperature: float = 0.8
    min_weight: float = 0.01

    # PKLemon-aligned knobs
    shrink_lambda: float = 0.28
    no_trade_band: float = 0.02

    # adaptive shrink
    turnover_gamma: float = 0.8
    shrink_cap: float = 0.85

    # sticky topN
    sticky_buffer: int = 2

    # caps + bounds
    cap_equity: float = 0.85
    cap_bond: float = 0.85
    cap_commodity: float = 0.40
    cap_lo: float = 0.0
    cap_hi: float = 1.0

    # risk_aversion bounds/base
    risk_aversion_base: float = 1.0
    risk_aversion_lo: float = 0.5
    risk_aversion_hi: float = 3.0

    # core_sat_spread bounds/base
    core_sat_spread_base: float = 0.0
    core_sat_spread_lo: float = -0.5
    core_sat_spread_hi: float = 0.5

    # topn bounds/base
    base_bond_n: int = 3
    base_commodity_n: int = 3
    topn_lo: int = 1
    topn_hi: int = 8

    # base class weights (public)
    class_weights: Dict[str, float] = None

    # warmup
    warmup_days: int = 0  # 0 => auto max(lookback)+5; warmup uses default allocation

    # audit
    audit_dir: str = "outputs/audit"

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

        router_cfg = RouterConfig(
            cap_lo=cfg.cap_lo, cap_hi=cfg.cap_hi,
            risk_aversion_lo=cfg.risk_aversion_lo, risk_aversion_hi=cfg.risk_aversion_hi,
            core_sat_spread_lo=cfg.core_sat_spread_lo, core_sat_spread_hi=cfg.core_sat_spread_hi,
            topn_lo=cfg.topn_lo, topn_hi=cfg.topn_hi,
            cap_equity=cfg.cap_equity, cap_bond=cfg.cap_bond, cap_commodity=cfg.cap_commodity,
            shrink_lambda=cfg.shrink_lambda,
            turnover_shrink_base=0.15,
            risk_aversion_base=cfg.risk_aversion_base,
            core_sat_spread_base=cfg.core_sat_spread_base,
            base_bond_n=cfg.base_bond_n, base_commodity_n=cfg.base_commodity_n,
        )
        self.router = PublicRuleRouter(router_cfg)
        self.audit = AuditLogger(audit_dir=cfg.audit_dir)

    def _pivot_close(self, prices: pd.DataFrame) -> pd.DataFrame:
        px = prices.pivot(index="date", columns="code", values="close").sort_index()
        return px.ffill()

    def _compute_features(self, px_close: pd.DataFrame, asof: pd.Timestamp) -> Dict[str, Any]:
        rets = px_close.pct_change()
        equity_mom = float(rets.mean(axis=1).rolling(20).sum().loc[asof]) if asof in rets.index else 0.0
        equity_vol = float(rets.mean(axis=1).rolling(20).std().loc[asof]) if asof in rets.index else 0.0
        if not np.isfinite(equity_mom): equity_mom = 0.0
        if not np.isfinite(equity_vol): equity_vol = 0.0
        return {"equity_mom": equity_mom, "equity_vol": equity_vol}

    def _score(self, px_close: pd.DataFrame, asof: pd.Timestamp, risk_aversion: float) -> pd.Series:
        mom = px_close.pct_change(self.cfg.lookback_mom).loc[asof]
        vol = px_close.pct_change().rolling(self.cfg.lookback_vol).std().loc[asof]
        common = mom.dropna().index.intersection(vol.dropna().index)
        if len(common) == 0:
            return pd.Series(dtype=float)
        mom, vol = mom.loc[common], vol.loc[common]
        return zscore(mom) - zscore(vol) * float(risk_aversion)

    def _apply_caps_and_core_sat(self, base_cls_w: Dict[str, float], caps: Dict[str, float], core_sat_spread: float):
        w_core = float(base_cls_w.get("EQUITY_CORE", 0.0))
        w_sat = float(base_cls_w.get("EQUITY_SAT", 0.0))
        w_bond = float(base_cls_w.get("BOND", 0.0))
        w_cmd = float(base_cls_w.get("COMMODITY", 0.0))

        w_eq = w_core + w_sat
        w_eq = min(w_eq, float(caps.get("EQUITY", w_eq)))
        w_bond = min(w_bond, float(caps.get("BOND", w_bond)))
        w_cmd = min(w_cmd, float(caps.get("COMMODITY", w_cmd)))

        s = w_eq + w_bond + w_cmd
        if s <= 1e-12:
            return {"EQUITY_CORE": 0.0, "EQUITY_SAT": 0.0, "BOND": 0.0, "COMMODITY": 0.0}, {"EQUITY": 0.0, "BOND": 0.0, "COMMODITY": 0.0}

        w_eq, w_bond, w_cmd = w_eq / s, w_bond / s, w_cmd / s

        sat_ratio = (w_sat / (w_core + w_sat + 1e-12))
        w_sat_eff = _clamp(sat_ratio - float(core_sat_spread), 0.0, 1.0)

        w_sat_new = w_eq * w_sat_eff
        w_core_new = w_eq * (1.0 - w_sat_eff)

        cls = {"EQUITY_CORE": w_core_new, "EQUITY_SAT": w_sat_new, "BOND": w_bond, "COMMODITY": w_cmd}
        asset = {"EQUITY": w_eq, "BOND": w_bond, "COMMODITY": w_cmd}
        return cls, asset

    def _equal_weight_within_class(self, universe: pd.DataFrame, cls_name: str, k: int) -> Dict[str, float]:
        codes = universe.loc[universe["asset_class"] == cls_name, "code"].astype(str).tolist()
        codes = sorted(codes)
        if not codes:
            return {}
        chosen = codes[:min(k, len(codes))]
        w = 1.0 / len(chosen)
        return {c: w for c in chosen}

    def _default_allocation(self, universe: pd.DataFrame, cls_w: Dict[str, float], topn_used: Dict[str, int]) -> Dict[str, float]:
        out = {}
        mapping = {
            "EQUITY_CORE": ("core", int(topn_used.get("core", 3))),
            "EQUITY_SAT": ("sat", int(topn_used.get("sat", 3))),
            "BOND": ("bond", int(topn_used.get("bond", 3))),
            "COMMODITY": ("commodity", int(topn_used.get("commodity", 3))),
        }
        for cls, (_, k) in mapping.items():
            bw = float(cls_w.get(cls, 0.0))
            if bw <= 0:
                continue
            ew = self._equal_weight_within_class(universe, cls, k=max(1, k))
            for code, ww in ew.items():
                out[code] = out.get(code, 0.0) + bw * ww
        return _renorm(out)

    def _apply_no_trade_band(self, w_target: Dict[str, float], w_prev: Dict[str, float], band: float) -> Dict[str, float]:
        keys = set(w_target.keys()) | set(w_prev.keys())
        out = {}
        for k in keys:
            wt = float(w_target.get(k, 0.0))
            wp = float(w_prev.get(k, 0.0))
            if abs(wt - wp) < band:
                wt = wp
            if wt > 0:
                out[k] = wt
        return _renorm(out)

    def _apply_shrink(self, w_band: Dict[str, float], w_prev: Dict[str, float], lam: float) -> Dict[str, float]:
        keys = set(w_band.keys()) | set(w_prev.keys())
        out = {}
        for k in keys:
            wb = float(w_band.get(k, 0.0))
            wp = float(w_prev.get(k, 0.0))
            wf = (1.0 - lam) * wb + lam * wp
            if wf > 0:
                out[k] = wf
        return _renorm(out)

    def _min_weight_cleanup_keep_prev(self, w: Dict[str, float], w_prev: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for k, v in w.items():
            if float(v) >= self.cfg.min_weight or k in w_prev:
                out[k] = float(v)
        return _renorm(out)

    def _rebalance_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if self.cfg.rebalance_mode == "month_end":
            return month_end_rebalance_days(dates, include_first=self.cfg.include_first_rebalance)
        return pd.DatetimeIndex(dates[:: self.cfg.rebalance_freq])

    def generate_weights(self, universe: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        start = pd.to_datetime(self.cfg.start)
        end = pd.to_datetime(self.cfg.end)

        px_close = self._pivot_close(prices)
        dates = px_close.index[(px_close.index >= start) & (px_close.index <= end)]
        rebalance_dates = self._rebalance_dates(dates)

        cls_map = dict(zip(universe["code"].astype(str), universe["asset_class"]))

        prev_final: Dict[str, float] = {}
        rows = []

        warmup_days = self.cfg.warmup_days or (max(self.cfg.lookback_mom, self.cfg.lookback_vol) + 5)
        # warmup 以交易日为单位：在 dates 里前 warmup_days 天都用默认仓位
        warmup_cut = dates[min(len(dates)-1, warmup_days-1)] if len(dates) > 0 and warmup_days > 0 else dates[0]

        for dt in rebalance_dates:
            asof_date = pd.to_datetime(dt).strftime("%Y-%m-%d")
            feat_dt = dt  # features 用 dt（内部特征已是rolling/shift型，这里是公开版）
            feats = self._compute_features(px_close, feat_dt)

            B = self.router.get_B_params(asof_date=asof_date, feats=feats)
            caps_used = B["caps"]
            base_lambda = float(B["shrink_lambda"])
            risk_aversion_used = float(B["risk_aversion"])
            core_sat_spread_used = float(B["core_sat_spread"])
            topn_used = B["topn"]

            cls_w_used, asset_w = self._apply_caps_and_core_sat(self.cfg.class_weights, caps_used, core_sat_spread_used)

            # --- warmup：默认仓位（只建仓一次，之后保持不变）---
            if dt <= warmup_cut:
                if not prev_final:
                    w_default = self._default_allocation(universe, cls_w_used, topn_used)
                    prev_final = dict(w_default)

                w_final = dict(prev_final)
                self.audit.log_rebalance(
                    asof_date=asof_date,
                    features=feats,
                    params_used={
                        "warmup_mode": "default_allocation",
                        "asset_caps": caps_used,
                        "core_sat_spread": core_sat_spread_used,
                        "topn": dict(topn_used),
                    },
                    selected={"core": [], "sat": [], "bond": [], "commodity": []},
                    asset_weights=asset_w,
                    target_weights=w_final,
                    final_weights=w_final,
                    audit_note={"summary": "warmup(default_allocation)", "bullets": []},
                    note="Warmup: strategic allocation only; weights held constant until lookback ready.",
                )
                for code, weight in w_final.items():
                    rows.append({"date": dt, "code": code, "weight": float(weight)})
                continue

            # --- active：score -> sticky topN -> within softmax -> target ---
            score = self._score(px_close, dt, risk_aversion=risk_aversion_used)

            selected = {"core": [], "sat": [], "bond": [], "commodity": []}
            sticky_info = {"core_retained": 0, "sat_retained": 0, "bond_retained": 0, "commodity_retained": 0}

            if score is None or score.empty:
                w_target = self._default_allocation(universe, cls_w_used, topn_used)
            else:
                w_target_series = pd.Series(0.0, index=px_close.columns)

                def prev_in_class(cls_name: str):
                    prev_codes = [c for c in prev_final.keys() if cls_map.get(str(c)) == cls_name]
                    return prev_codes

                for cls, base_w in cls_w_used.items():
                    if base_w <= 0:
                        continue
                    cls_codes = [c for c in score.index if cls_map.get(str(c)) == cls]
                    if not cls_codes:
                        continue

                    cls_scores = score.loc[cls_codes].sort_values(ascending=False)
                    ranked = [str(x) for x in cls_scores.index.tolist()]

                    if cls == "EQUITY_CORE":
                        k = int(topn_used.get("core", 3))
                        picked = sticky_topn(ranked, prev_in_class(cls), k, buffer=self.cfg.sticky_buffer)
                        sticky_info["core_retained"] = len(set(picked) & set(prev_in_class(cls)))
                        selected["core"] = picked
                    elif cls == "EQUITY_SAT":
                        k = int(topn_used.get("sat", 3))
                        picked = sticky_topn(ranked, prev_in_class(cls), k, buffer=self.cfg.sticky_buffer)
                        sticky_info["sat_retained"] = len(set(picked) & set(prev_in_class(cls)))
                        selected["sat"] = picked
                    elif cls == "BOND":
                        k = int(topn_used.get("bond", 3))
                        picked = sticky_topn(ranked, prev_in_class(cls), k, buffer=self.cfg.sticky_buffer)
                        sticky_info["bond_retained"] = len(set(picked) & set(prev_in_class(cls)))
                        selected["bond"] = picked
                    else:
                        k = int(topn_used.get("commodity", 3))
                        picked = sticky_topn(ranked, prev_in_class(cls), k, buffer=self.cfg.sticky_buffer)
                        sticky_info["commodity_retained"] = len(set(picked) & set(prev_in_class(cls)))
                        selected["commodity"] = picked

                    chosen = cls_scores.loc[picked]
                    w_cls = softmax(chosen, temp=self.cfg.within_temperature) * float(base_w)
                    w_target_series.loc[w_cls.index] = w_cls

                w_target_series[w_target_series.abs() < self.cfg.min_weight] = 0.0
                w_target = {str(k): float(v) for k, v in w_target_series[w_target_series > 0].items()}
                w_target = _renorm(w_target)

            # --- no-trade band (先) ---
            w_band = self._apply_no_trade_band(w_target, prev_final, band=float(self.cfg.no_trade_band))

            # --- adaptive shrink (后) ---
            tpred = turnover_pred_l1(w_band, prev_final)
            lam = adaptive_shrink_lambda(base_lambda, tpred, gamma=float(self.cfg.turnover_gamma), cap=float(self.cfg.shrink_cap))
            w_shrunk = self._apply_shrink(w_band, prev_final, lam=lam)

            # --- min_weight cleanup（保留旧仓）---
            w_final = self._min_weight_cleanup_keep_prev(w_shrunk, prev_final)

            params_used = {
                "asset_caps": caps_used,
                "shrink_lambda_base": base_lambda,
                "turnover_pred": float(tpred),
                "turnover_gamma": float(self.cfg.turnover_gamma),
                "shrink_lambda_adaptive": float(lam),
                "risk_aversion": float(risk_aversion_used),
                "core_sat_spread": float(core_sat_spread_used),
                "topn": dict(topn_used),
                "sticky_buffer": int(self.cfg.sticky_buffer),
                "sticky_retained": sticky_info,
                "no_trade_band": float(self.cfg.no_trade_band),
                "min_weight": float(self.cfg.min_weight),
            }

            self.audit.log_rebalance(
                asof_date=asof_date,
                features=feats,
                params_used=params_used,
                selected=selected,
                asset_weights=asset_w,
                target_weights=w_target,
                final_weights=w_final,
                audit_note={"summary": f"active_model=True, regime={B.get('regime','neutral')}, conf={B.get('confidence',0.0):.2f}", "bullets": B.get("rationale", [])},
                note="Active: month-end rebalance; sticky topN; no-trade band then adaptive shrink; keep-prev min_weight cleanup.",
            )

            for code, weight in w_final.items():
                rows.append({"date": dt, "code": code, "weight": float(weight)})
            prev_final = dict(w_final)

        return pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)
