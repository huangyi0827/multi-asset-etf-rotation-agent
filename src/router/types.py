from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class BParams:
    """
    对齐 PKLemon.py 中 B 参数调度器输出字段：
    confidence/regime/caps/shrink_lambda/risk_aversion/core_sat_spread/topn/raw
    """
    confidence: float = 0.0
    regime: str = "neutral"  # risk_on | neutral | risk_off
    caps: Dict[str, float] = field(default_factory=lambda: {"EQUITY": 0.85, "BOND": 0.85, "COMMODITY": 0.40})
    shrink_lambda: float = 0.28
    risk_aversion: float = 1.0
    core_sat_spread: float = 0.0
    topn: Dict[str, int] = field(default_factory=lambda: {"core": 3, "sat": 3, "bond": 3, "commodity": 3})
    rationale: List[str] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None  # 公开版保留字段，但通常为 None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": float(self.confidence),
            "regime": str(self.regime),
            "caps": dict(self.caps),
            "shrink_lambda": float(self.shrink_lambda),
            "risk_aversion": float(self.risk_aversion),
            "core_sat_spread": float(self.core_sat_spread),
            "topn": dict(self.topn),
            "rationale": list(self.rationale),
            "raw": self.raw,
        }
