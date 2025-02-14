from typing import Iterable

import numpy as np
import pandas as pd
from lru import LRU

from analyzer.commons import Value, determine_bit_depth, BIT_DEPTH

BOOL_INDEX_CACHE_SIZE: int = 200


class IndexLocations:
    def __init__(self, index: 'Index', locations: np.ndarray, cache_key=None):
        self.index: Index = index
        self._cache_key = cache_key
        self.count: int
        self._locations: np.ndarray
        if locations.dtype == bool:
            self.count = locations.sum()
            if self.count >= self.index.directly_store_bool_index_threshold:
                self._locations = locations
            else:
                self._locations = np.array(np.nonzero(locations), dtype=np.uint32)
        elif locations.dtype == np.uint32:
            self.count = len(locations)
            if self.count >= self.index.directly_store_bool_index_threshold:
                loc_bool: np.ndarray = np.zeros(self.index.total_count, dtype=bool)
                loc_bool[locations] = 1
                self._locations = loc_bool
            else:
                self._locations = locations
        else:
            raise Exception('Unexpected index dtype: ', self._locations.dtype)

    def _get_bool_index(self):
        if self._locations.dtype == bool:
            return self._locations
        elif self._locations.dtype == np.uint32:
            if self._cache_key is not None and self._cache_key in self.index.bool_index_cache:
                return self.index.bool_index_cache[self._cache_key]
            else:
                loc_bool: np.ndarray = np.zeros(self.index.total_count, dtype=bool)
                loc_bool[self._locations] = 1
                if self._cache_key is not None:
                    self.index.bool_index_cache[self._cache_key] = loc_bool
                return loc_bool
        else:
            raise Exception('Unexpected index dtype: ', self._locations.dtype)

    def __and__(self, other: 'IndexLocations') -> 'IndexLocations':
        bool_idx_self: np.ndarray = self._get_bool_index()
        bool_idx_other: np.ndarray = other._get_bool_index()
        new_bool_idx: np.ndarray = bool_idx_self & bool_idx_other
        return IndexLocations(self.index, new_bool_idx)

    def __or__(self, other: 'IndexLocations') -> 'IndexLocations':
        bool_idx_self: np.ndarray = self._get_bool_index()
        bool_idx_other: np.ndarray = other._get_bool_index()
        new_bool_idx: np.ndarray = bool_idx_self | bool_idx_other
        return IndexLocations(self.index, new_bool_idx)

    def __invert__(self) -> 'IndexLocations':
        bool_idx_self: np.ndarray = self._get_bool_index()
        new_loc: np.ndarray = ~bool_idx_self
        return IndexLocations(self.index, new_loc, None)


class Index:

    def __init__(self, data_df: pd.DataFrame, column_types: dict[str, str], target_column: str, target_value: Value):
        self.directly_store_bool_index_threshold: int = int(len(data_df) * 1 / (8 if BIT_DEPTH == 64 else 4))
        self._index: dict[str, dict[Value, IndexLocations]]
        self._init_index(data_df, column_types, target_column, target_value)
        self.target_column: str = target_column
        self.target_value: Value = target_value
        self.total_count = len(data_df)
        self.total_error_locations: IndexLocations = self.get_locations(target_column, target_value)
        self.total_error_count: int = self.total_error_locations.count
        self.total_error_rate = self.total_error_locations.count / self.total_count
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
        self.bool_index_cache = LRU(BOOL_INDEX_CACHE_SIZE)

    @staticmethod
    def _get_cache_key(column: str, value: Value | pd.Interval) -> str:
        return column + '=' + str(value)

    def _init_index(self, data_df: pd.DataFrame, column_types: dict[str, str], target_column: str, target_value: Value):
        col_indexes: dict[str, dict[Value, IndexLocations]] = {}  # ndarray of bool/np.uint32
        print('Start indexing...')
        for col_name in data_df.columns:
            col_indexes[col_name] = {}

        data_array: np.ndarray = data_df.to_numpy(copy=False)
        for col_pos in range(0, len(data_df.columns)):
            col_name: str = data_df.columns[col_pos]
            is_float_col: bool = column_types[col_name] == 'float'
            unique_values: pd.Series = data_df[col_name].unique()
            print('Indexing %s, unique values: %d' % (col_name, len(unique_values)))
            if len(unique_values) <= 400:
                for val in unique_values:
                    val: Value | pd.Interval
                    if is_float_col and issubclass(type(val), float) and np.isnan(val):
                        non_na_loc: np.ndarray = data_df[col_name].isna().to_numpy(copy=False)
                        col_indexes[col_name][val] = IndexLocations(self, non_na_loc, self._get_cache_key(col_name, val))
                    else:
                        val_loc: np.ndarray = (data_df[col_name] == val).to_numpy(copy=False)
                        col_indexes[col_name][val] = IndexLocations(self, val_loc, self._get_cache_key(col_name, val))
            else:
                is_float_col: bool = column_types[col_name] == 'float'
                col_index = {}
                for val in data_df[col_name].unique():
                    if is_float_col and issubclass(type(val), float) and np.isnan(val):
                        val = np.nan
                    col_index[val] = []
                for row_num in range(0, len(data_array)):
                    val: Value | pd.Interval = data_array[row_num][col_pos]
                    if is_float_col and issubclass(type(val), float) and np.isnan(val):
                        val = np.nan
                    col_index[val].append(row_num)
                for val, non_zero_list in col_index.items():
                    val: Value | pd.Interval
                    non_zero_list: list[int]
                    col_indexes[col_name][val] = IndexLocations(self, np.array(non_zero_list, dtype=np.uint32),
                                                                self._get_cache_key(col_name, val))
        self._index = col_indexes

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

    def get_locations(self, column: str, value: Value | pd.Interval) -> IndexLocations:
        return self._index[column][value]

