from typing import TypeAlias

Value: TypeAlias = str | float | int


def calc_weight(dimensions: int, error_coverage: float, error_rate: float, total_error_rate: float) -> float:
    # if error_rate <= total_error_rate:
    #     return 0
    # return dimensions * error_coverage * (error_rate - total_error_rate)
    return dimensions**2 * error_coverage**0.5 * error_rate**2
