import random
from typing import Iterable

import numpy as np
import pandas as pd


class Index:

    def __init__(self, data_df: pd.DataFrame, threshold: int):
        self._index: dict[str, dict[str, pd.Series]]
        self._select_vector: dict[str, list[(str, int)]]
        self._init_index(data_df, threshold)
        self._init_select_vector()
        self.total_count = len(data_df)
        filtered_columns: list[str] = []
        col: str
        for col in data_df.columns:
            if col.startswith("Unnamed:"):
                continue
            elif col not in self._index:  # Not reach threshold
                continue
            else:
                filtered_columns.append(col)
        self.columns: list[str] = filtered_columns

    def _init_index(self, data_df: pd.DataFrame, threshold: int):
        col_names: list[str] = data_df.columns
        col_count: int = len(col_names)
        col_indexes: list[dict[str, np.ndarray]] = [{} for _ in range(0, col_count)]
        for col_pos in range(0, col_count):
            col_name: str = col_names[col_pos]
            for val in data_df[col_name].unique():
                col_indexes[col_pos][val] = np.zeros(len(data_df), dtype=np.bool)

        df_values = data_df.values
        for row_num in range(0, len(df_values)):
            row = df_values[row_num]
            for col_pos in range(0, col_count):
                val: str = row[col_pos]
                col_indexes[col_pos][val][row_num] = True

        index: dict[str, dict[str, pd.Series]] = {}
        for col_pos in range(0, col_count):
            for val in col_indexes[col_pos].keys():
                val: str
                val_index: np.ndarray = col_indexes[col_pos][val]
                if val_index.sum() < threshold:
                    continue
                col_name: str = col_names[col_pos]
                if col_name not in index:
                    index[col_name] = {}
                index[col_name][val] = pd.Series(val_index)
        self._index = index

    def _init_select_vector(self):
        self._select_vector = {}  # column -> (value, cumulative_freq)
        for col in self._index.keys():
            val_count_map: dict[str, int] = {}
            col_index: dict[str, pd.Series] = self._index[col]
            for key in col_index:
                val = key
                val_index: pd.Series = col_index[key]
                val_count_map[val] = val_index.sum()

            total = 0
            cumulate_list: list[(str, int)] = []
            for key in val_count_map:
                val = key
                count = val_count_map[val]
                total += count
                cumulate_list.append((val, total))
            self._select_vector[col] = cumulate_list

    def get_columns_after(self, column: str | None):
        if column is None:
            return self.columns
        # TODO 二分查找
        pos = self.columns.index(column)
        if pos == len(self.columns) - 1:
            return []
        else:
            return self.columns[pos+1:]

    def get_columns_before(self, column: str | None):
        if column is None:
            return None
        pos = self.columns.index(column)
        if pos == 0:
            return []
        else:
            return self.columns[:pos]

    def get_values_by_column(self, column: str) -> Iterable:
        return self._index[column].keys()

    def get_locations(self, column: str, value: str) -> pd.Series:
        return self._index[column][value]

    def random_select_value_by_freq(self, col: str) -> str:
        """
        :param col: column name
        :return: selected value
        """
        col_select_vec = self._select_vector[col]
        total = col_select_vec[len(col_select_vec)-1][1]
        rand = random.randint(0, total)
        low = 0
        high = len(col_select_vec) - 1
        while low < high:
            mid: int = int((low + high) / 2)
            cur: int = col_select_vec[mid][1]
            if cur < rand:
                if mid == low:
                    low += 1
                else:
                    low = mid
            elif cur > rand:
                if mid == high:
                    high -= 1
                else:
                    high = mid
            else:  # cur == rand
                return col_select_vec[mid][0]
        select_idx = high
        return col_select_vec[select_idx][0]

