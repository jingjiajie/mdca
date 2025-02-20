import numpy as np
import pandas as pd

from analyzer.Index import Index, IndexLocations
from analyzer.commons import Value, calc_weight


class ResultItem:

    def __init__(self, column: str, value: Value | pd.Interval, locations: IndexLocations):
        self.column: str = column
        self.value: Value | pd.Interval = value
        self.locations: IndexLocations = locations

    def __str__(self):
        return f"{self.column}={self.value}"

    def __eq__(self, other: 'ResultItem'):
        return self.column == other.column and self.value == other.value


class CalculatedResult:

    def __init__(self, count: int, error_count: int, error_coverage: float, error_rate: float, weight: float):
        self.count: int = count
        self.error_count: int = error_count
        self.error_rate: float = error_rate
        self.error_coverage: float = error_coverage
        self.weight: float = weight


class ResultPath:

    def __init__(self, items: list[ResultItem], locations: IndexLocations):
        self.items: list[ResultItem] = items
        self.locations: IndexLocations | None = locations

    def __str__(self):
        item_str_list: list[str] = []
        for item in self.items:
            item_str_list.append(str(item))
        return "[" + ", ".join(item_str_list) + "]"

    def __getitem__(self, column: str) -> ResultItem | None:
        for item in self.items:
            if item.column == column:
                return item
        return None

    def calculate(self, index: Index) -> CalculatedResult:
        result_items: list[ResultItem] = self.items
        total_error_loc: IndexLocations = index.get_locations(index.target_column, index.target_value)
        total_error_count: int = total_error_loc.count
        if len(result_items) == 0:
            return CalculatedResult(index.total_count, total_error_count,  1, index.total_error_rate,
                                    calc_weight(0, 1, index.total_error_rate, index.total_error_rate))
        count: int = self.locations.count
        error_count: int = (self.locations & total_error_loc).count
        error_rate: float = error_count / count
        error_coverage: float = error_count / total_error_count
        return CalculatedResult(count, error_count, error_coverage, error_rate,
                                calc_weight(len(self.items), error_coverage, error_rate, index.total_error_rate))
