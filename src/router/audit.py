from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

def _sanitize(x):
    # JSON strict: convert NaN/Inf -> None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize(v) for v in x]
    return x

@dataclass
class AuditLogger:
    audit_dir: str = "outputs/audit"
    filename: str = "audit.jsonl"

    def _path(self) -> Path:
        d = Path(self.audit_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d / self.filename

    def log_rebalance(
        self,
        asof_date: str,
        features: Dict[str, Any],
        params_used: Dict[str, Any],
        selected: Dict[str, Any],
        asset_weights: Dict[str, Any],
        target_weights: Dict[str, Any],
        final_weights: Dict[str, Any],
        audit_note: Optional[Dict[str, Any]] = None,
        note: str = "",
    ) -> None:
        row = {
            "asof_date": asof_date,
            "features": features,
            "params_used": params_used,
            "selected": selected,
            "asset_weights": asset_weights,
            "target_weights": target_weights,
            "final_weights": final_weights,
            "audit_note": audit_note or {},
            "note": note,
        }
        row = _sanitize(row)
        p = self._path()
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
