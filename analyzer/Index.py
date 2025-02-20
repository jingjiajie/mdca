import threading
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd

from analyzer.commons import Value

# TODO 改成比例
ROW_NUMBER_INTERSECT_THRESHOLD: int = 160000
ROW_NUMBER_FAST_INTERCEPT_COUNT_THRESHOLD: int = 500000

_thread_local: threading.local = threading.local()

class IndexLocationType(Enum):
    BOOL = 0
    ROW_NUMBER = 1


class IndexLocations:
    def __init__(self, index: 'Index', locations: np.ndarray):
        self.index: Index = index
        self.count: int
        self.best_type: IndexLocationType
        self._locations_bool: np.ndarray | None = None
        self._locations_row_number: np.ndarray | None = None

        if locations.dtype == bool:
            self.count = np.count_nonzero(locations)
            if self.count < self.index.directly_store_bool_index_threshold:
                self.best_type = IndexLocationType.ROW_NUMBER
                self._locations_row_number = np.array(np.nonzero(locations)[0], dtype=np.uint32)
            else:
                self.best_type = IndexLocationType.BOOL
                self._locations_bool = locations
        elif locations.dtype == np.uint32:
            self.count = len(locations)
            if self.count >= self.index.directly_store_bool_index_threshold:
                self.best_type = IndexLocationType.BOOL
                loc_bool: np.ndarray = np.zeros(self.index.total_count, dtype=bool)
                loc_bool[locations] = 1
                self._locations_bool = loc_bool
            else:
                self.best_type = IndexLocationType.ROW_NUMBER
                self._locations_row_number = locations
        else:
            raise Exception('Unexpected index dtype: ', locations.dtype)

    def cache(self, index_type: IndexLocationType):
        if index_type == IndexLocationType.BOOL:
            self._get_bool_index(calculate_if_need=True, cache_calculated_result=True)
        elif index_type == IndexLocationType.ROW_NUMBER:
            self._get_row_number_index(calculate_if_need=True, cache_calculated_result=True)

    def clear_cache(self):
        if self.best_type == IndexLocationType.BOOL:
            self._locations_row_number = None
        elif self.best_type == IndexLocationType.ROW_NUMBER:
            self._locations_bool = None

    def _get_bool_index(self, calculate_if_need: bool, cache_calculated_result: bool) -> np.ndarray | None:
        if self._locations_bool is not None:
            return self._locations_bool
        elif calculate_if_need:
            if self._locations_row_number is not None:
                loc_bool: np.ndarray = np.zeros(self.index.total_count, dtype=bool)
                loc_bool[self._locations_row_number] = 1
                if cache_calculated_result:
                    self._locations_bool = loc_bool
                return loc_bool
        return None

    def _get_row_number_index(self, calculate_if_need: bool, cache_calculated_result: bool) -> np.ndarray | None:
        if self._locations_row_number is not None:
            return self._locations_row_number
        elif calculate_if_need:
            if self._locations_bool is not None:
                loc_row_number: np.ndarray = np.array(np.nonzero(self._locations_bool)[0], dtype=np.uint32)
                if cache_calculated_result:
                    self._locations_row_number = loc_row_number
                return loc_row_number
        return None

    def fast_intersect_count(self, other: 'IndexLocations') -> int | None:
        if (self.count >= ROW_NUMBER_FAST_INTERCEPT_COUNT_THRESHOLD and
                other.count >= ROW_NUMBER_FAST_INTERCEPT_COUNT_THRESHOLD):
            return None
        loc_bool: np.ndarray | None = None
        loc_row_number: np.ndarray | None = None

        if (self.count < ROW_NUMBER_FAST_INTERCEPT_COUNT_THRESHOLD and
                other.count < ROW_NUMBER_FAST_INTERCEPT_COUNT_THRESHOLD):
            if (self._locations_row_number is not None and self._locations_bool is not None
                    and other._locations_row_number is not None and other._locations_bool is not None):
                if self.count < other.count:
                    loc_row_number = self._locations_row_number
                    loc_bool = other._locations_bool
                else:
                    loc_row_number = other._locations_row_number
                    loc_bool = self._locations_bool
            else:
                if self._locations_row_number is not None and other._locations_bool is not None:
                    loc_row_number = self._locations_row_number
                    loc_bool = other._locations_bool
                elif self._locations_bool is not None and other._locations_row_number is not None:
                    loc_row_number = other._locations_row_number
                    loc_bool = self._locations_bool

        if loc_bool is None or loc_row_number is None:
            for a, b in [(self, other), (other, self)]:
                loc_bool = None
                loc_row_number = None
                if b.count >= ROW_NUMBER_FAST_INTERCEPT_COUNT_THRESHOLD:  # a.count < threshold
                    if a._locations_row_number is not None and b._locations_bool is not None:
                        loc_row_number = a._locations_row_number
                        loc_bool = b._locations_bool
                if loc_bool is not None and loc_row_number is not None:
                    break

        if loc_bool is None or loc_row_number is None:
            return None
        selected_vec: np.ndarray = loc_bool[loc_row_number]
        return np.count_nonzero(selected_vec)

    def __and__(self, other: 'IndexLocations') -> 'IndexLocations':
        if self.count + other.count < ROW_NUMBER_INTERSECT_THRESHOLD:
            loc_self: np.ndarray = self._get_row_number_index(calculate_if_need=True, cache_calculated_result=False)
            loc_other: np.ndarray = other._get_row_number_index(calculate_if_need=True, cache_calculated_result=False)
            new_loc: np.uint32 = np.intersect1d(loc_self, loc_other, assume_unique=True)
            return IndexLocations(self.index, new_loc)
        else:
            bool_idx_self: np.ndarray = self._get_bool_index(calculate_if_need=True, cache_calculated_result=False)
            bool_idx_other: np.ndarray = other._get_bool_index(calculate_if_need=True, cache_calculated_result=False)
            new_bool_idx: np.ndarray = bool_idx_self & bool_idx_other
            return IndexLocations(self.index, new_bool_idx)

    def __or__(self, other: 'IndexLocations') -> 'IndexLocations':
        bool_idx_self: np.ndarray = self._get_bool_index(calculate_if_need=True, cache_calculated_result=False)
        bool_idx_other: np.ndarray = other._get_bool_index(calculate_if_need=True, cache_calculated_result=False)
        new_bool_idx: np.ndarray = bool_idx_self | bool_idx_other
        return IndexLocations(self.index, new_bool_idx)

    def __invert__(self) -> 'IndexLocations':
        bool_idx_self: np.ndarray = self._get_bool_index(calculate_if_need=True, cache_calculated_result=False)
        new_loc: np.ndarray = ~bool_idx_self
        return IndexLocations(self.index, new_loc)


class Index:

    def __init__(self, data_df: pd.DataFrame, column_types: dict[str, str], target_column: str, target_value: Value):
        self.directly_store_bool_index_threshold: int = int(len(data_df) * 1 / 4)  # sizeof(np.uint32)
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
                        col_indexes[col_name][val] = IndexLocations(self, non_na_loc)
                    else:
                        val_loc: np.ndarray = (data_df[col_name] == val).to_numpy(copy=False)
                        col_indexes[col_name][val] = IndexLocations(self, val_loc)
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
                    col_indexes[col_name][val] = IndexLocations(self, np.array(non_zero_list, dtype=np.uint32))
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

    @staticmethod
    def fast_predict_intersect_count_less_than(
            location_list: list[IndexLocations], min_intersect_count: int, sample_rate: int = 0.01) -> bool:
        bool_index_list: list[np.ndarray] = []
        for loc in location_list:
            if loc._locations_bool is None:
                raise Exception('Only bool index is supported!')
            elif len(loc._locations_bool) != len(location_list[0]._locations_bool):
                raise Exception('Input bool indexes must have same length!')
            bool_index_list.append(loc._locations_bool)
        length: int = len(bool_index_list[0])
        total_sample_count: int = int(length * sample_rate)
        tmp_index_key: str = "temp_bool_index_" + str(total_sample_count)
        if not hasattr(_thread_local, tmp_index_key):
            _thread_local.__setattr__(tmp_index_key, np.ndarray(total_sample_count, dtype=bool))
        sampled_intersection: np.ndarray = _thread_local.__getattribute__(tmp_index_key)
        sampled_intersection[:] = 1
        for bool_index in bool_index_list:
            sampled: np.ndarray = np.lib.stride_tricks.as_strided(
                bool_index, shape=(total_sample_count,), strides=(int(1/sample_rate),), writeable=False)
            sampled_intersection &= sampled
        sampled_nonzero_count: int = np.count_nonzero(sampled_intersection)
        estimated_total_nonzero_count: int = int(sampled_nonzero_count / sample_rate)

        # Index.pred += 1
        # cur_loc: IndexLocations = location_list[0]
        # for loc in location_list:
        #     cur_loc = cur_loc & loc
        # if estimated_total_nonzero_count / min_intersect_count < 0.8:
        #     if cur_loc.count >= min_intersect_count:
        #         Index.pred_wrong += 1
        #         print("### PREDICT WRONG! pred: %d, wrong: %d, sample_estimate: %d, real_count: %d" %
        #               (Index.pred, Index.pred_wrong, estimated_total_nonzero_count, cur_loc.count))

        if estimated_total_nonzero_count / min_intersect_count < 0.8:
            return True
        else:
            return False


