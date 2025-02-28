import threading
import time
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd
from bitarray import bitarray

from analyzer.commons import Value

BOOL_FAST_PREDICT_INTERSECT_COUNT_SAMPLE_RATE: float = 0.01

_thread_local: threading.local = threading.local()


class IndexLocationType(Enum):
    BOOL = 0
    ROW_NUMBER = 1


class IndexLocations:
    def __init__(self, locations: bitarray):
        self._count: int = -1
        self._locations = locations
        self.index_length = len(locations)

        sample_step: int = int(1/BOOL_FAST_PREDICT_INTERSECT_COUNT_SAMPLE_RATE)
        self._sampled_locations: bitarray = locations[sample_step-1::sample_step]

        sample_length: int = int(len(locations) * BOOL_FAST_PREDICT_INTERSECT_COUNT_SAMPLE_RATE)
        tmp_index_key: str = "temp_bitarray_" + str(sample_length)
        if not hasattr(_thread_local, tmp_index_key):
            _thread_local.__setattr__(tmp_index_key, bitarray(sample_length))
        self._temp_sample_intersect_buff: bitarray = _thread_local.__getattribute__(tmp_index_key)

    @property
    def count(self) -> int:
        if self._count == -1:
            self._count = self._locations.count(1)
        return self._count

    @staticmethod
    def fast_predict_bool_intersect_count(loc_list: list['IndexLocations']) -> int | None:
        sampled_intersection: bitarray = loc_list[0]._temp_sample_intersect_buff
        sampled_intersection[:] = 1
        for loc in loc_list:
            sampled_intersection &= loc._sampled_locations
        sampled_nonzero_count: int = sampled_intersection.count(1)
        estimated_total_nonzero_count: int = int(sampled_nonzero_count / BOOL_FAST_PREDICT_INTERSECT_COUNT_SAMPLE_RATE)
        return estimated_total_nonzero_count

    def __and__(self, other: 'IndexLocations') -> 'IndexLocations':
        new_bit_idx: bitarray = self._locations & other._locations
        return IndexLocations(new_bit_idx)

    def __iand__(self, other: 'IndexLocations') -> 'IndexLocations':
        self._count = -1
        self._locations &= other._locations
        return self

    def __or__(self, other: 'IndexLocations') -> 'IndexLocations':
        new_bit_idx: bitarray = self._locations | other._locations
        return IndexLocations(new_bit_idx)

    def __ior__(self, other: 'IndexLocations') -> 'IndexLocations':
        self._count = -1
        self._locations |= other._locations
        return self

    def __invert__(self) -> 'IndexLocations':
        new_bit_index: bitarray = ~self._locations
        return IndexLocations(new_bit_index)

    def __copy__(self) -> 'IndexLocations':
        copied = IndexLocations(bitarray.copy(self._locations))
        copied._count = self.count
        return copied


class Index:

    def __init__(self, data_df: pd.DataFrame, target_column: str, target_value: Value):
        self.total_count = len(data_df)
        self._index: dict[str, dict[Value, IndexLocations]]
        self._init_index(data_df)
        self.target_column: str = target_column
        self.target_value: Value = target_value
        self.total_target_locations: IndexLocations = self.get_locations(target_column, target_value)
        self.total_target_count: int = self.total_target_locations.count
        self.total_target_rate = self.total_target_locations.count / self.total_count

        filtered_columns: list[str] = []
        col: str
        for col in data_df.columns:
            if col.startswith("Unnamed:"):
                continue
            elif col not in self._index:
                continue
            elif col == target_column:
                continue
            else:
                filtered_columns.append(col)
        self.non_target_columns: list[str] = filtered_columns

    def _init_index(self, data_df: pd.DataFrame):
        print("Indexing data...")
        start = time.time()
        col_indexes: dict[str, dict[Value, IndexLocations]] = {}  # ndarray of bool/np.uint32
        for col_name in data_df.columns:
            col_indexes[col_name] = {}

        data_array: np.ndarray = data_df.to_numpy(copy=False)
        for col_pos in range(0, len(data_df.columns)):
            col_name: str = data_df.columns[col_pos]
            unique_values: pd.Series = pd.Series(data_df[col_name].unique())
            print(' - Indexing %s, unique values: %d' % (col_name, len(unique_values)))
            if len(unique_values) <= 400:
                for val in unique_values:
                    val: Value | pd.Interval
                    if val is None or issubclass(type(val), float) and np.isnan(val):
                        na_loc_bool: np.ndarray = data_df[col_name].isna().to_numpy(copy=False)
                        na_loc_bit: bitarray = bitarray(buffer=np.packbits(na_loc_bool).data)
                        na_loc_bit = na_loc_bit[:len(data_df)]
                        if val is None:
                            col_indexes[col_name][None] = IndexLocations(na_loc_bit)
                        else:
                            col_indexes[col_name][np.nan] = IndexLocations(na_loc_bit)
                    else:
                        val_loc_bool: np.ndarray = (data_df[col_name] == val).to_numpy(copy=False)
                        val_loc_bit: bitarray = bitarray(buffer=np.packbits(val_loc_bool).data)
                        val_loc_bit = val_loc_bit[:len(data_df)]
                        col_indexes[col_name][val] = IndexLocations(val_loc_bit)
            else:
                col_index: dict[Value, list[int]] = {}
                for val in data_df[col_name].unique():
                    col_index[val] = []

                for row_num in range(0, len(data_array)):
                    val: Value | pd.Interval = data_array[row_num][col_pos]
                    if val is None:
                        col_index[None].append(row_num)
                    elif issubclass(type(val), float) and np.isnan(val):
                        col_index[np.nan].append(row_num)
                    else:
                        col_index[val].append(row_num)

                for val, row_number_list in col_index.items():
                    val: Value | pd.Interval
                    row_number_list: list[int]
                    loc_bit: bitarray = bitarray(len(data_df))
                    loc_bit[row_number_list] = 1
                    col_indexes[col_name][val] = IndexLocations(loc_bit)
        print("Index data cost: %.2f seconds" % (time.time() - start))
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



