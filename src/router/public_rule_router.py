from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

from .types import BParams

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))

@dataclass
class RouterConfig:
    # caps bounds
    cap_lo: float = 0.0
    cap_hi: float = 1.0

    # risk_aversion bounds
    risk_aversion_lo: float = 0.5
    risk_aversion_hi: float = 3.0

    # core_sat_spread bounds
    core_sat_spread_lo: float = -0.5
    core_sat_spread_hi: float = 0.5

    # topn bounds
    topn_lo: int = 1
    topn_hi: int = 8

    # base values (对齐 StrategyConfig 命名)
    cap_equity: float = 0.85
    cap_bond: float = 0.85
    cap_commodity: float = 0.40

    shrink_lambda: float = 0.28
    turnover_shrink_base: float = 0.15

    risk_aversion_base: float = 1.0
    core_sat_spread_base: float = 0.0

    base_bond_n: int = 3
    base_commodity_n: int = 3

class PublicRuleRouter:
    """
    公开版：用规则模拟“B: 参数调度器”，但输出字段与 PKLemon 保持一致：
    confidence/regime/caps/shrink_lambda/risk_aversion/core_sat_spread/topn/rationale
    """
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg

    def get_B_params(self, asof_date: str, feats: Dict[str, Any]) -> Dict[str, Any]:
        """
        feats 建议包含（公开版可简化）：
        - equity_mom: float (近N日收益)
        - equity_vol: float (近N日波动)
        """
        mom = float(feats.get("equity_mom", 0.0) or 0.0)
        vol = float(feats.get("equity_vol", 0.0) or 0.0)

        # 1) regime
        if mom < -0.02 or vol > 0.018:
            regime = "risk_off"
        elif mom > 0.02 and vol < 0.012:
            regime = "risk_on"
        else:
            regime = "neutral"

        # 2) confidence：越“极端”越自信（公开版启发式）
        conf = _clamp(abs(mom) / 0.05 + max(0.0, (vol - 0.012) / 0.02), 0.0, 1.0)

        # 3) caps（asset_caps）
        caps = {
            "EQUITY": _clamp(self.cfg.cap_equity, self.cfg.cap_lo, self.cfg.cap_hi),
            "BOND": _clamp(self.cfg.cap_bond, self.cfg.cap_lo, self.cfg.cap_hi),
            "COMMODITY": _clamp(self.cfg.cap_commodity, self.cfg.cap_lo, self.cfg.cap_hi),
        }

        rationale = []

        if regime == "risk_off":
            # 更保守：压权益上限、提高债券上限、略提高换手抑制
            caps["EQUITY"] = _clamp(caps["EQUITY"] - 0.15 * conf, self.cfg.cap_lo, self.cfg.cap_hi)
            caps["BOND"] = _clamp(caps["BOND"] + 0.10 * conf, self.cfg.cap_lo, self.cfg.cap_hi)
            caps["COMMODITY"] = _clamp(caps["COMMODITY"] + 0.05 * conf, self.cfg.cap_lo, self.cfg.cap_hi)
            rationale.append("Regime=risk_off: reduce EQUITY cap, increase BOND/COMMODITY caps")
        elif regime == "risk_on":
            caps["EQUITY"] = _clamp(caps["EQUITY"] + 0.05 * conf, self.cfg.cap_lo, self.cfg.cap_hi)
            rationale.append("Regime=risk_on: allow slightly higher EQUITY cap")
        else:
            rationale.append("Regime=neutral: keep caps near baseline")

        # 4) shrink_lambda（换手抑制强度）
        shrink = _clamp(self.cfg.shrink_lambda + 0.20 * conf * (1.0 if regime != "risk_on" else 0.5), 0.0, 1.0)

        # 5) risk_aversion（越大越保守）
        ra = self.cfg.risk_aversion_base
        if regime == "risk_off":
            ra = _clamp(ra + 1.2 * conf, self.cfg.risk_aversion_lo, self.cfg.risk_aversion_hi)
            rationale.append("Increase risk_aversion in risk_off")
        elif regime == "risk_on":
            ra = _clamp(ra - 0.3 * conf, self.cfg.risk_aversion_lo, self.cfg.risk_aversion_hi)
            rationale.append("Slightly decrease risk_aversion in risk_on")
        else:
            ra = _clamp(ra, self.cfg.risk_aversion_lo, self.cfg.risk_aversion_hi)

        # 6) core_sat_spread（正=更偏 CORE，负=更偏 SAT）
        sp = _clamp(self.cfg.core_sat_spread_base, self.cfg.core_sat_spread_lo, self.cfg.core_sat_spread_hi)
        if regime == "risk_off":
            sp = _clamp(sp + 0.20 * conf, self.cfg.core_sat_spread_lo, self.cfg.core_sat_spread_hi)
            rationale.append("Positive core_sat_spread in risk_off (prefer CORE)")
        elif regime == "risk_on":
            sp = _clamp(sp - 0.10 * conf, self.cfg.core_sat_spread_lo, self.cfg.core_sat_spread_hi)
            rationale.append("Negative core_sat_spread in risk_on (allow more SAT)")

        # 7) topn（核心/卫星/债/商品）
        topn = {
            "core": int(_clamp(3, self.cfg.topn_lo, self.cfg.topn_hi)),
            "sat": int(_clamp(3, self.cfg.topn_lo, self.cfg.topn_hi)),
            "bond": int(_clamp(self.cfg.base_bond_n + (1 if regime == "risk_off" else 0), self.cfg.topn_lo, self.cfg.topn_hi)),
            "commodity": int(_clamp(self.cfg.base_commodity_n + (1 if regime == "risk_on" else 0), self.cfg.topn_lo, self.cfg.topn_hi)),
        }

        out = BParams(
            confidence=float(conf),
            regime=regime,
            caps=caps,
            shrink_lambda=float(shrink),
            risk_aversion=float(ra),
            core_sat_spread=float(sp),
            topn=topn,
            rationale=rationale[:5],
            raw=None,
        )
        return out.to_dict()
