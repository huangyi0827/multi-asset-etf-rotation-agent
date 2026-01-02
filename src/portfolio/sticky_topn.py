from __future__ import annotations
from typing import Iterable, List, Sequence, Set

def sticky_topn(
    ranked_codes: Sequence[str],
    prev_holdings: Iterable[str],
    topn: int,
    buffer: int = 2
) -> List[str]:
    """
    ranked_codes: 本期按分数从高到低排序的代码
    prev_holdings: 上期该桶（class）持仓代码
    topn: 需要选的数量
    buffer: sticky buffer，允许保留落在 topn+buffer 内的上期持仓
    """
    ranked = [str(x) for x in ranked_codes]
    prev = [str(x) for x in prev_holdings]
    if topn <= 0 or len(ranked) == 0:
        return []

    # 在 topn+buffer 内的上期持仓优先保留（按原排名顺序）
    cutoff = min(len(ranked), topn + max(0, buffer))
    eligible: Set[str] = set(ranked[:cutoff])

    retained = [c for c in ranked if (c in eligible and c in prev)]
    retained = retained[:topn]

    # 再按排名补足
    picked = list(retained)
    for c in ranked:
        if len(picked) >= topn:
            break
        if c not in picked:
            picked.append(c)

    return picked
