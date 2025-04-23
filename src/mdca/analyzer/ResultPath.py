import numpy as np

from mdca.analyzer.Index import Index, IndexLocations
from mdca.analyzer.commons import Value, calc_weight_fairness, calc_weight_error


class BinResultValue:

    def __init__(self, lower_cut_point_id: int | None, lower_cut_point_value: float | None,
                 upper_cut_point_id: int | None, upper_cut_point_value: float | None):
        self.lower_cut_point_id: int | None = lower_cut_point_id
        self.lower_cut_point_value: float | None = lower_cut_point_value
        self.upper_cut_point_id: int | None = upper_cut_point_id
        self.upper_cut_point_value: float | None = upper_cut_point_value


class ResultItem:

    def __init__(self, column: str, column_type: str, value: Value | BinResultValue, locations: IndexLocations):
        self.column: str = column
        self.column_type: str = column_type
        self.value: Value = value
        self.locations: IndexLocations = locations

    # def to_json(self):
    #     d: dict = dict(self.__dict__)
    #     if isinstance(d['value'], float) and np.isnan(d['value']):
    #         d['value'] = 'NaN'
    #     del d['locations']
    #     del d['column_type']
    #     return d

    def __str__(self):
        if isinstance(self.value, BinResultValue):
            if self.value.upper_cut_point_value is None:
                return f"{self.column}>={self._get_value_str(self.column_type, self.value.lower_cut_point_value)}"
            elif self.value.lower_cut_point_value is None:
                return f"{self.column}<{self._get_value_str(self.column_type, self.value.upper_cut_point_value)}"
            else:
                return (f"{self._get_value_str(self.column_type, self.value.lower_cut_point_value)}<="
                        f"{self.column}<"
                        f"{self._get_value_str(self.column_type, self.value.upper_cut_point_value)}")
        else:
            return f"{self.column}={self._get_value_str(self.column_type, self.value)}"

    def __eq__(self, other: 'ResultItem'):
        return self.column == other.column and self.value == other.value

    @staticmethod
    def _get_value_str(column_type: str, value: Value) -> str:
        if np.issubdtype(type(value), float) and np.isnan(value):
            return 'nan'
        elif column_type == 'int':
            return str(int(value))
        elif column_type == 'float':
            return '%.2f' % value
        else:
            return str(value)


class ResultPath:

    def __init__(self, items: list[ResultItem], locations: IndexLocations, search_mode: str):
        self.items: list[ResultItem] = items
        self.locations: IndexLocations = locations
        self.search_mode: str = search_mode

    # def to_json(self) -> dict:
    #     d: dict = dict(self.__dict__)
    #     item_dict_list: list[dict] = []
    #     for item in self.items:
    #         item_dict_list.append(item.to_json())
    #     d['items'] = item_dict_list
    #     del d['locations']
    #     return d

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
        column_values: dict[str, Value] = {}
        for item in self.items:
            column_values[item.column] = item.value
        if len(self.items) == 0:
            if self.search_mode == 'fairness':
                return CalculatedResult(self, index.total_count, 1, 1,
                                        index.total_target_count, 1, index.total_target_rate,
                                        calc_weight_fairness(0, 1,
                                                             index.total_target_rate, index.total_target_rate))
            elif self.search_mode == 'error':
                return CalculatedResult(self, index.total_count, 1, 1,
                                        index.total_target_count, 1, index.total_target_rate,
                                        calc_weight_error(0, 1,
                                                          index.total_target_rate, index.total_target_rate))
        count: int = self.locations.count
        coverage: float = count / index.total_count
        target_count: int = -1
        target_rate: float = -1
        target_coverage: float = -1
        if index.target_column is not None:
            total_target_loc: IndexLocations = index.get_categorical_locations(index.target_column, index.target_value)
            total_target_count: int = total_target_loc.count
            target_count = (self.locations & total_target_loc).count
            target_rate = target_count / count
            target_coverage = target_count / total_target_count
        weight: float = -1
        baseline_coverage: float = -1
        if self.search_mode == 'error':
            weight = calc_weight_error(len(self.items), target_coverage, target_rate, index.total_target_rate)
        elif self.search_mode == 'fairness':
            weight = calc_weight_fairness(len(self.items), coverage, target_rate, index.total_target_rate)
        return CalculatedResult(self, count, coverage, baseline_coverage, target_count, target_coverage,
                                target_rate, weight)


class CalculatedResult(ResultPath):

    def __init__(self, result_path: ResultPath, count: int,
                 coverage: float, baseline_coverage: float, target_count: int,
                 target_coverage: float, target_rate: float, weight: float, ):
        super().__init__(result_path.items, result_path.locations, result_path.search_mode)
        self.count: int = count
        self.coverage: float = coverage
        self.baseline_coverage: float = baseline_coverage
        self.target_count: int = target_count
        self.target_rate: float = target_rate
        self.target_coverage: float = target_coverage
        self.weight: float = weight
