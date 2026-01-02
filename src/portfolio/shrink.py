from __future__ import annotations
from typing import Dict, Tuple

def turnover_pred_l1(w_target: Dict[str, float], w_prev: Dict[str, float]) -> float:
    """
    预测换手：0.5 * L1 距离（常见定义）
    """
    keys = set(w_target.keys()) | set(w_prev.keys())
    l1 = 0.0
    for k in keys:
        l1 += abs(float(w_target.get(k, 0.0)) - float(w_prev.get(k, 0.0)))
    return 0.5 * l1

def adaptive_shrink_lambda(base_lambda: float, turnover_pred: float, gamma: float, cap: float) -> float:
    """
    lambda = min(cap, base + gamma * turnover_pred)
    """
    lam = float(base_lambda) + float(gamma) * float(turnover_pred)
    if lam < 0.0:
        lam = 0.0
    if lam > float(cap):
        lam = float(cap)
    return lam
