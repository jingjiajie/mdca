import numpy as np
import pandas as pd

from analyzer.Index import Index, IndexLocations
from analyzer.commons import Value, calc_weight_fairness, calc_weight_distribution


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
            if np.issubdtype(type(self.value), float) and not np.isnan(self.value):
                return str(int(self.value))
        return str(self.value)


class ResultPath:

    def __init__(self, items: list[ResultItem], locations: IndexLocations, search_mode: str):
        self.items: list[ResultItem] = items
        self.locations: IndexLocations = locations
        self.search_mode: str = search_mode

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

    def calculate(self, index: Index) -> 'CalculatedResult':
        if isinstance(self, CalculatedResult):
            return self
        column_values: dict[str, Value | pd.Interval] = {}
        for item in self.items:
            column_values[item.column] = item.value
        baseline_coverage: float = index.get_column_combination_coverage_baseline(column_values)
        if len(self.items) == 0:
            if self.search_mode == 'fairness':
                total_target_loc: IndexLocations = index.get_locations(index.target_column, index.target_value)
                total_target_count: int = total_target_loc.count
                return CalculatedResult(self, index.total_count, 1, baseline_coverage, total_target_count,
                                        1, index.total_target_rate,
                                        calc_weight_fairness(0, 1,
                                                             index.total_target_rate, index.total_target_rate))
            elif self.search_mode == 'distribution':
                return CalculatedResult(self, index.total_count, 1, 1, -1, -1,
                                        calc_weight_distribution(0, 1,1))
        count: int = self.locations.count
        if self.search_mode == 'fairness':
            total_target_loc: IndexLocations = index.get_locations(index.target_column, index.target_value)
            total_target_count: int = total_target_loc.count
            target_count: int = (self.locations & total_target_loc).count
            target_rate: float = target_count / count
            target_coverage: float = target_count / total_target_count
            return CalculatedResult(self, count, count/index.total_count, baseline_coverage, target_count, target_coverage, target_rate,
                                    calc_weight_fairness(len(self.items), target_coverage, target_rate,
                                                         index.total_target_rate))
        elif self.search_mode == 'distribution':
            coverage: float = count / index.total_count
            return CalculatedResult(self, count, coverage, baseline_coverage, -1, -1, -1,
                                    calc_weight_distribution(len(self.items), coverage, baseline_coverage)
                                    )


class CalculatedResult(ResultPath):

    def __init__(self, result_path: ResultPath, count: int,
                 total_coverage: float, baseline_coverage: float, target_count: int,
                 target_coverage: float, target_rate: float, weight: float, ):
        super().__init__(result_path.items, result_path.locations, result_path.search_mode)
        self.count: int = count
        self.total_coverage: float = total_coverage
        self.baseline_coverage: float = baseline_coverage
        self.target_count: int = target_count
        self.target_rate: float = target_rate
        self.target_coverage: float = target_coverage
        self.weight: float = weight

