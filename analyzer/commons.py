import sys
from typing import TypeAlias

Value: TypeAlias = str | float | int | bool | None

_ALPHA: float = 1
_BETA: float = 1/2
_GAMMA: float = 3/2
# _EPSILON: float = 0.001


def calc_weight(dimensions: int, error_coverage: float, error_rate: float, total_error_rate: float) -> float:
    # if error_rate < total_error_rate:
    #     return dimensions ** _ALPHA * error_coverage ** _BETA * (error_rate*_EPSILON) ** _GAMMA
    # else:
    #     return dimensions ** _ALPHA * error_coverage ** _BETA * (error_rate - total_error_rate + _EPSILON) ** _GAMMA

    return _ALPHA**-(dimensions-1) * error_coverage**_BETA * max(error_rate - total_error_rate, 0)**_GAMMA
