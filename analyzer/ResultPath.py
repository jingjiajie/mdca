import numpy as np
import pandas as pd

from analyzer.Index import Index, IndexLocations
from analyzer.commons import Value, calc_weight


class ResultItem:

    def __init__(self, column: str, column_type: str, value: Value | pd.Interval, locations: IndexLocations):
        self.column: str = column
        self.column_type: str = column_type
        self.value: Value | pd.Interval = value
        self.locations: IndexLocations = locations

    def __str__(self):
        return f"{self.column}={self._get_value_str()}"

    def __eq__(self, other: 'ResultItem'):
        return self.column == other.column and self.value == other.value

    def _get_value_str(self) -> str:
        if self.column_type == 'int':
            if np.issubdtype(type(self.value), float):
                return str(int(self.value))
        return str(self.value)


class CalculatedResult:

    def __init__(self, count: int, target_count: int, target_coverage: float, target_rate: float, weight: float):
        self.count: int = count
        self.target_count: int = target_count
        self.target_rate: float = target_rate
        self.target_coverage: float = target_coverage
        self.weight: float = weight


class ResultPath:

    def __init__(self, items: list[ResultItem], locations: IndexLocations):
        self.items: list[ResultItem] = items
        self.locations: IndexLocations = locations

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
        total_target_loc: IndexLocations = index.get_locations(index.target_column, index.target_value)
        total_target_count: int = total_target_loc.count
        if len(result_items) == 0:
            return CalculatedResult(index.total_count, total_target_count,  1, index.total_target_rate,
                                    calc_weight(0, 1, index.total_target_rate, index.total_target_rate))
        count: int = self.locations.count
        target_count: int = (self.locations & total_target_loc).count
        target_rate: float = target_count / count
        target_coverage: float = target_count / total_target_count
        return CalculatedResult(count, target_count, target_coverage, target_rate,
                                calc_weight(len(self.items), target_coverage, target_rate, index.total_target_rate))
