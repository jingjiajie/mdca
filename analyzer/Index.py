import random
from typing import Iterable

import numpy as np
import pandas as pd


class Index:

    def __init__(self, data_df: pd.DataFrame, target_column: str, target_value: str, min_error_coverage: float):
        self._index: dict[str, dict[str, pd.Series]]
        self._init_index(data_df, target_column, target_value, min_error_coverage)
        self.target_column: str = target_column
        self.target_value: str = target_value
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

    def _init_index(self, data_df: pd.DataFrame, target_column: str, target_value: str, min_error_coverage: float):
        col_names: list[str] = data_df.columns
        col_count: int = len(col_names)
        col_indexes: list[dict[str, np.ndarray]] = [{} for _ in range(0, col_count)]
        target_col_val_index: np.ndarray | None = None
        for col_pos in range(0, col_count):
            col_name: str = col_names[col_pos]
            for val in data_df[col_name].unique():
                col_indexes[col_pos][val] = np.zeros(len(data_df), dtype=np.bool)
                if col_name == target_column and val == target_value:
                    target_col_val_index = col_indexes[col_pos][val]

        df_values = data_df.values
        for row_num in range(0, len(df_values)):
            row = df_values[row_num]
            for col_pos in range(0, col_count):
                val: str = row[col_pos]
                col_indexes[col_pos][val][row_num] = True

        error_count: int = target_col_val_index.sum()
        index: dict[str, dict[str, pd.Series]] = {}
        for col_pos in range(0, col_count):
            for val in col_indexes[col_pos].keys():
                val: str
                val_index: np.ndarray = col_indexes[col_pos][val]
                col_name: str = col_names[col_pos]
                if col_name != target_column and (val_index & target_col_val_index).sum() / error_count < min_error_coverage:
                    continue
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

    def get_values_by_column(self, column: str) -> Iterable:
        return self._index[column].keys()

    def get_locations(self, column: str, value: str) -> pd.Series:
        return self._index[column][value]

