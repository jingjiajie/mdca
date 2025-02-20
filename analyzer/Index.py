import threading
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd

from analyzer.commons import Value

ROW_NUMBER_INTERSECT_THRESHOLD: float = 0.02
ROW_NUMBER_FAST_INTERSECT_COUNT_THRESHOLD: float = 0.1
ROW_NUMBER_UNION_THRESHOLD: float = 0.1
BOOL_FAST_PREDICT_INTERSECT_COUNT_SAMPLE_RATE: float = 0.01

_thread_local: threading.local = threading.local()


class IndexLocationType(Enum):
    BOOL = 0
    ROW_NUMBER = 1


class IndexLocations:
    def __init__(self, locations: np.ndarray, index_length: int):
        self.count: int
        self.index_length: int = index_length
        self.best_type: IndexLocationType
        self._locations_bool: np.ndarray | None = None
        self._locations_row_number: np.ndarray | None = None
        self._sampled_locations_bool: np.ndarray | None = None

        sample_length: int = int(index_length * BOOL_FAST_PREDICT_INTERSECT_COUNT_SAMPLE_RATE)
        tmp_index_key: str = "temp_bool_index_" + str(sample_length)
        if not hasattr(_thread_local, tmp_index_key):
            _thread_local.__setattr__(tmp_index_key, np.ndarray(sample_length, dtype=bool))
        self._temp_sample_intersect_buff: np.ndarray = _thread_local.__getattribute__(tmp_index_key)

        directly_store_bool_index_threshold: int = int(index_length / 4)  # sizeof(np.uint32)
        if locations.dtype == bool:
            self.count = np.count_nonzero(locations)
            if self.count < directly_store_bool_index_threshold:
                self.best_type = IndexLocationType.ROW_NUMBER
                self._locations_row_number = np.array(np.nonzero(locations)[0], dtype=np.uint32)
            else:
                self.best_type = IndexLocationType.BOOL
                self._locations_bool = locations
        elif locations.dtype == np.uint32:
            self.count = len(locations)
            if self.count >= directly_store_bool_index_threshold:
                self.best_type = IndexLocationType.BOOL
                loc_bool: np.ndarray = np.zeros(self.index_length, dtype=bool)
                loc_bool[locations] = 1
                self._locations_bool = loc_bool
            else:
                self.best_type = IndexLocationType.ROW_NUMBER
                self._locations_row_number = locations
        else:
            raise Exception('Unexpected index dtype: ', locations.dtype)

    @property
    def nbytes(self):
        byte_count: int = 0
        if self._locations_bool is not None:
            byte_count += self._locations_bool.nbytes
        if self._locations_row_number is not None:
            byte_count += self._locations_row_number.nbytes
        if self._sampled_locations_bool is not None:
            byte_count += self._sampled_locations_bool.nbytes
        return byte_count

    def cache(self, index_type: IndexLocationType):
        if index_type == IndexLocationType.BOOL:
            self._get_bool_index(calculate_if_need=True, cache_calculated_result=True)
            self._get_sampled_bool_index(calculate_if_need=True, cache_calculated_result=True)
        elif index_type == IndexLocationType.ROW_NUMBER:
            self._get_row_number_index(calculate_if_need=True, cache_calculated_result=True)

    def clear_cache(self):
        if self.best_type == IndexLocationType.BOOL:
            self._locations_row_number = None
            self._sampled_locations_bool = None
        elif self.best_type == IndexLocationType.ROW_NUMBER:
            self._locations_bool = None
            self._sampled_locations_bool = None

    def _get_sampled_bool_index(self, calculate_if_need: bool, cache_calculated_result: bool) -> np.ndarray | None:
        loc_bool: np.ndarray = self._get_bool_index(calculate_if_need, cache_calculated_result)
        if loc_bool is None:
            return None
        if self._sampled_locations_bool is not None:
            return self._sampled_locations_bool
        elif calculate_if_need:
            total_sample_count: int = int(len(self._locations_bool) * BOOL_FAST_PREDICT_INTERSECT_COUNT_SAMPLE_RATE)
            if total_sample_count < 100:
                return None
            sampled: np.ndarray = np.lib.stride_tricks.as_strided(
                                        loc_bool,
                                        shape=(total_sample_count,),
                                        strides=(int(1/BOOL_FAST_PREDICT_INTERSECT_COUNT_SAMPLE_RATE),),
                                        writeable=False)
            if cache_calculated_result:
                self._sampled_locations_bool = sampled
            return sampled
        return None

    def _get_bool_index(self, calculate_if_need: bool, cache_calculated_result: bool) -> np.ndarray | None:
        if self._locations_bool is not None:
            return self._locations_bool
        elif calculate_if_need:
            if self._locations_row_number is not None:
                loc_bool: np.ndarray = np.zeros(self.index_length, dtype=bool)
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
        if (self.count >= ROW_NUMBER_FAST_INTERSECT_COUNT_THRESHOLD * self.index_length and
                other.count >= ROW_NUMBER_FAST_INTERSECT_COUNT_THRESHOLD * self.index_length):
            return None
        loc_bool: np.ndarray | None = None
        loc_row_number: np.ndarray | None = None

        if (self.count < ROW_NUMBER_FAST_INTERSECT_COUNT_THRESHOLD * self.index_length and
                other.count < ROW_NUMBER_FAST_INTERSECT_COUNT_THRESHOLD * self.index_length):
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
                if b.count >= ROW_NUMBER_FAST_INTERSECT_COUNT_THRESHOLD * self.index_length:  # a.count < threshold
                    if a._locations_row_number is not None and b._locations_bool is not None:
                        loc_row_number = a._locations_row_number
                        loc_bool = b._locations_bool
                if loc_bool is not None and loc_row_number is not None:
                    break

        if loc_bool is None or loc_row_number is None:
            return None
        selected_vec: np.ndarray = loc_bool[loc_row_number]
        return np.count_nonzero(selected_vec)

    @staticmethod
    def fast_predict_bool_intersect_count(loc_list: list['IndexLocations']) -> int | None:
        sample_loc_list: list[np.ndarray] = []
        for loc in loc_list:
            sample_loc: np.ndarray = loc._get_sampled_bool_index(calculate_if_need=False,
                                                                 cache_calculated_result=False)
            if sample_loc is None:
                return None
            sample_loc_list.append(sample_loc)
            if len(sample_loc_list[0]) != len(sample_loc):
                raise Exception('Sampled bool indexes must have same length! actual: %d, %d' %
                                (len(sample_loc_list[0]), len(sample_loc)))
        sampled_intersection: np.ndarray = loc_list[0]._temp_sample_intersect_buff
        sampled_intersection[:] = 1
        for sample_loc in sample_loc_list:
            sampled_intersection &= sample_loc
        sampled_nonzero_count: int = np.count_nonzero(sampled_intersection)
        estimated_total_nonzero_count: int = int(sampled_nonzero_count / BOOL_FAST_PREDICT_INTERSECT_COUNT_SAMPLE_RATE)
        return estimated_total_nonzero_count

    def __and__(self, other: 'IndexLocations') -> 'IndexLocations':
        if self.count + other.count < ROW_NUMBER_INTERSECT_THRESHOLD * self.index_length:
            loc_self: np.ndarray = self._get_row_number_index(calculate_if_need=True, cache_calculated_result=False)
            loc_other: np.ndarray = other._get_row_number_index(calculate_if_need=True, cache_calculated_result=False)
            new_loc: np.ndarray = np.intersect1d(loc_self, loc_other, assume_unique=True)
            return IndexLocations(new_loc, self.index_length)
        else:
            bool_idx_self: np.ndarray = self._get_bool_index(calculate_if_need=True, cache_calculated_result=False)
            bool_idx_other: np.ndarray = other._get_bool_index(calculate_if_need=True, cache_calculated_result=False)
            new_bool_idx: np.ndarray = bool_idx_self & bool_idx_other
            return IndexLocations(new_bool_idx, self.index_length)

    def __or__(self, other: 'IndexLocations') -> 'IndexLocations':
        if self.count + other.count < ROW_NUMBER_UNION_THRESHOLD * self.index_length:
            row_number_index_self: np.ndarray =\
                self._get_row_number_index(calculate_if_need=True, cache_calculated_result=False)
            row_number_index_other: np.ndarray =\
                other._get_row_number_index(calculate_if_need=True, cache_calculated_result=False)
            new_loc: np.ndarray = np.union1d(row_number_index_self, row_number_index_other)
            return IndexLocations(new_loc, self.index_length)
        else:
            bool_idx_self: np.ndarray = self._get_bool_index(calculate_if_need=True, cache_calculated_result=False)
            bool_idx_other: np.ndarray = other._get_bool_index(calculate_if_need=True, cache_calculated_result=False)
            new_bool_idx: np.ndarray = bool_idx_self | bool_idx_other
            return IndexLocations(new_bool_idx, self.index_length)

    def __invert__(self) -> 'IndexLocations':
        bool_idx_self: np.ndarray = self._get_bool_index(calculate_if_need=True, cache_calculated_result=False)
        new_loc: np.ndarray = ~bool_idx_self
        return IndexLocations(new_loc, self.index_length)


class Index:

    def __init__(self, data_df: pd.DataFrame, column_types: dict[str, str], target_column: str, target_value: Value):
        self.total_count = len(data_df)
        self._index: dict[str, dict[Value, IndexLocations]]
        self._init_index(data_df, column_types, target_column, target_value)
        self.target_column: str = target_column
        self.target_value: Value = target_value
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

    @property
    def nbytes(self):
        byte_count: int = 0
        for col in self._index.keys():
            for val in self._index[col].keys():
                loc: IndexLocations = self._index[col][val]
                byte_count += loc.nbytes
        return byte_count

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
                        col_indexes[col_name][val] = IndexLocations(non_na_loc, self.total_count)
                    else:
                        val_loc: np.ndarray = (data_df[col_name] == val).to_numpy(copy=False)
                        col_indexes[col_name][val] = IndexLocations(val_loc, self.total_count)
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
                    col_indexes[col_name][val] = (
                        IndexLocations(np.array(non_zero_list, dtype=np.uint32), self.total_count))
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
    def fast_predict_bool_intersect_count(loc_list: list['IndexLocations']) -> int | None:
        return IndexLocations.fast_predict_bool_intersect_count(loc_list)



