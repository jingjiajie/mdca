import sys
from typing import TypeAlias

Value: TypeAlias = str | float | int | bool | None

_ALPHA: float = 1
_BETA: float = 1/2
_GAMMA: float = 3/2
# _EPSILON: float = 0.001


def calc_weight(dimensions: int, target_coverage: float, target_rate: float, total_target_rate: float) -> float:
    # if target_rate < total_target_rate:
    #     return dimensions ** _ALPHA * target_coverage ** _BETA * (target_rate*_EPSILON) ** _GAMMA
    # else:
    #     return dimensions ** _ALPHA * target_coverage ** _BETA * (target_rate - total_target_rate + _EPSILON) ** _GAMMA

    return _ALPHA**-(dimensions-1) * target_coverage**_BETA * max(target_rate - total_target_rate, 0)**_GAMMA
