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


def calc_weight_error(dimensions: int, error_coverage: float, error_rate: float, total_error_rate: float) -> float:
    ALPHA: float = 1
    BETA: float = 1 / 2
    GAMMA: float = 3 / 2
    return ALPHA**-(dimensions-1) * error_coverage**BETA * abs(error_rate - total_error_rate)**GAMMA


def calc_weight_fairness(dimensions: int, coverage: float, target_rate: float, total_target_rate: float) -> float:
    ALPHA: float = 1
    BETA: float = 1 / 2
    GAMMA: float = 3 / 2
    return ALPHA**-(dimensions-1) * coverage**BETA * abs(target_rate - total_target_rate)**GAMMA


def calc_weight_distribution(dimensions: int, coverage: float, baseline_coverage: float) -> float:
    return (10000 *
            2**-(dimensions - 1) *
            (coverage - baseline_coverage)**2 *
            max(coverage / baseline_coverage, baseline_coverage / coverage)**2)
