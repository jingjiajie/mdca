from typing import Iterable

import numpy as np
import pandas as pd

from analyzer.commons import Value


class Index:

    def __init__(self, data_df: pd.DataFrame, column_types: dict[str, str], target_column: str, target_value: Value):
        self._index: dict[str, dict[str, pd.Series]]
        self._init_index(data_df, column_types, target_column, target_value)
        self.target_column: str = target_column
        self.target_value: Value = target_value
        self.total_count = len(data_df)
        self.total_error_rate = self.get_locations(target_column, target_value).sum() / self.total_count
        filtered_columns: list[str] = []
        col: str
        for col in data_df.columns:
            if col.startswith("Unnamed:"):
                continue
            elif col not in self._index:  # Not reach threshold
                continue
            elif col == target_column:
                continue
            else:
                filtered_columns.append(col)
        self.non_target_columns: list[str] = filtered_columns

    def _init_index(self, data_df: pd.DataFrame, column_types: dict[str, str], target_column: str, target_value: Value):
        col_names: list[str] = data_df.columns
        col_indexes: dict[str, dict[Value, np.ndarray]] = {}
        target_col_val_index: np.ndarray | None = None
        for col_name in col_names:
            is_float_col: bool = column_types[col_name] == 'float'
            col_indexes[col_name] = {}
            for val in data_df[col_name].unique():
                # todo 检查issubclass()的性能！
                if is_float_col and issubclass(type(val), float) and np.isnan(val):
                    val = np.nan
                col_indexes[col_name][val] = np.zeros(len(data_df), dtype=np.bool)
                if col_name == target_column and (val == target_value or val is target_value):  # np.nan is np.nan
                    target_col_val_index = col_indexes[col_name][val]

        for col_name in col_names:
            series_array: np.ndarray = data_df[col_name].values
            is_float_col: bool = column_types[col_name] == 'float'
            col_index = col_indexes[col_name]
            for row_num in range(0, len(series_array)):
                val: Value = series_array[row_num]
                # todo 检查issubclass()的性能！
                if is_float_col and issubclass(type(val), float) and np.isnan(val):
                    val = np.nan
                col_index[val][row_num] = True

        index: dict[str, dict[str, pd.Series]] = {}
        for col_name in col_names:
            for val in col_indexes[col_name].keys():
                val: Value
                val_index: np.ndarray = col_indexes[col_name][val]
                if col_name not in index:
                    index[col_name] = {}
                index[col_name][val] = pd.Series(val_index)
        self._index = index

    def get_columns_after(self, column: str | None):
        if column is None:
            return self.non_target_columns
        # TODO 二分查找
        pos = self.non_target_columns.index(column)
        if pos == len(self.non_target_columns) - 1:
            return []
        else:
            return self.non_target_columns[pos + 1:]

    def get_values_by_column(self, column: str) -> Iterable[Value | pd.Interval]:
        return self._index[column].keys()

    def get_locations(self, column: str, value: Value | pd.Interval) -> pd.Series:
        return self._index[column][value]

