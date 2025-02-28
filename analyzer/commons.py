from typing import TypeAlias

Value: TypeAlias = str | float | int | bool | None


class ColumnInfo:
    def __init__(self, column: str, col_type: str, binning: bool,
                 q00: float | None, q01: float | None, q99: float | None, q100: float | None):
        self.column: str = column
        self.column_type: str = col_type
        self.binning: bool = binning
        self.q00: float | None = q00
        self.q01: float | None = q01
        self.q99: float | None = q99
        self.q100: float | None = q100


def calc_weight_fairness(dimensions: int, target_coverage: float,
                         target_rate: float, total_target_rate: float) -> float:
    _ALPHA: float = 1
    _BETA: float = 1 / 2
    _GAMMA: float = 3 / 2
    # if target_rate < total_target_rate:
    #     return dimensions ** _ALPHA * target_coverage ** _BETA * (target_rate*_EPSILON) ** _GAMMA
    # else:
    #     return dimensions ** _ALPHA * target_coverage ** _BETA * (target_rate - total_target_rate + _EPSILON) ** _GAMMA

    return _ALPHA**-(dimensions-1) * target_coverage**_BETA * abs(target_rate - total_target_rate)**_GAMMA


def calc_weight_distribution(dimensions: int, coverage: float, baseline_coverage: float) -> float:
    # if coverage < baseline_coverage:
    #     return 0
    return (10000 *
            2**-(dimensions - 1) *
            (coverage - baseline_coverage)**2 *
            max(coverage / baseline_coverage, baseline_coverage / coverage)**2)
