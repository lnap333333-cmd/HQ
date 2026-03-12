"""C1: Elite-guided migration."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..encoding import Encoding
from ..instance import ProblemInstance


def apply(
    instance: ProblemInstance,
    source: List[Encoding],
    target: List[Encoding],
    rate: float = 0.1,
    target_algo: Optional[str] = None,
) -> Tuple[List[Encoding], List[Encoding]]:
    if not source or not target:
        return source, target

    # HHO 作为 C1 接收方时：更高迁移率 + 多猎物池（多目标精英）以强化探索
    if target_algo == "hho":
        rate = max(rate, 0.2)
        migrants = _select_prey_pool(instance, source, target, rate)
    else:
        k = max(1, int(len(target) * rate))
        src_sorted = sorted(source, key=lambda e: sum(instance.evaluate(e)))
        migrants = src_sorted[:k]

    k = len(migrants)
    migrants = src_sorted[:k]
    tgt_sorted = sorted(target, key=lambda e: sum(instance.evaluate(e)), reverse=True)
    new_target = migrants + tgt_sorted[k:]
    return source, new_target


def _select_prey_pool(
    instance: ProblemInstance,
    source: List[Encoding],
    target: List[Encoding],
    rate: float,
) -> List[Encoding]:
    """为 HHO 选择多猎物池：f1/f2/f3 各取精英，增强探索多样性。"""
    k = max(1, int(len(target) * rate))
    objs = [instance.evaluate(e) for e in source]
    seen: set[int] = set()
    migrants: List[Encoding] = []

    # 各目标最优各取若干，形成多猎物池
    per_obj = max(1, k // 3)
    for obj_idx in range(3):
        sorted_idx = sorted(range(len(objs)), key=lambda i: objs[i][obj_idx])
        for idx in sorted_idx:
            if idx not in seen and len(migrants) < k:
                seen.add(idx)
                migrants.append(source[idx])
            if len([j for j in sorted_idx if j in seen]) >= per_obj:
                break

    # 不足时用综合最优补足
    src_sorted = sorted(source, key=lambda e: sum(instance.evaluate(e)))
    for e in src_sorted:
        if len(migrants) >= k:
            break
        if e not in migrants:
            migrants.append(e)
    return migrants[:k]

