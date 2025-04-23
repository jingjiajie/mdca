from typing import TypeAlias

Value: TypeAlias = str | float | int | bool | None


class ColumnInfo:
    def __init__(self, column: str, col_type: str, binning: bool):
        self.column: str = column
        self.column_type: str = col_type
        self.binning: bool = binning
        self.cut_points: list[float | int] | None = None
        self.min: float | int | None = None
        self.max: float | int | None = None


def calc_weight_error(dimensions: int, error_coverage: float, error_rate: float, total_error_rate: float) -> float:
    ALPHA: float = 1
    BETA: float = 1 / 2
    # BETA: float = 1
    # GAMMA: float = 3 / 2
    GAMMA: float = 2
    if error_rate < total_error_rate:
        return 0
    return ALPHA**(dimensions-1) * error_coverage**BETA * abs(error_rate - total_error_rate)**GAMMA


def calc_weight_fairness(dimensions: int, coverage: float, target_rate: float, total_target_rate: float) -> float:
    ALPHA: float = 1
    BETA: float = 1 / 2
    GAMMA: float = 3 / 2
    if target_rate < total_target_rate:
        return 0
    return ALPHA**-(dimensions-1) * coverage**BETA * abs(target_rate - total_target_rate)**GAMMA
