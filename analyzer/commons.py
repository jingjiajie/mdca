import sys
from typing import TypeAlias

Value: TypeAlias = str | float | int

_ALPHA: float = 0
_BETA: float = 1/2
_GAMMA: float = 1
_EPSILON: float = 0.001


def calc_weight(dimensions: int, error_coverage: float, error_rate: float, total_error_rate: float) -> float:
    if error_rate < total_error_rate:
        return dimensions ** _ALPHA * error_coverage ** _BETA * (error_rate*_EPSILON) ** _GAMMA
    else:
        return dimensions ** _ALPHA * error_coverage ** _BETA * (error_rate - total_error_rate + _EPSILON) ** _GAMMA
    # return dimensions**_ALPHA * error_coverage**_BETA * error_rate**_GAMMA