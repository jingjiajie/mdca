import threading
import time
from typing import Iterable, cast

import numpy as np
import pandas as pd
from bitarray import bitarray

from mdca.analyzer.commons import Value, ColumnInfo

BOOL_FAST_PREDICT_INTERSECT_COUNT_THRESHOLD: int = 10000
BOOL_FAST_PREDICT_INTERSECT_COUNT_SAMPLE_RATE: float = 0.01

_thread_local: threading.local = threading.local()


class IndexLocations:
    def __init__(self, locations: bitarray):
        self._count: int = -1
        self._locations = locations
        self.index_length = len(locations)

        sample_step: int = int(1 / BOOL_FAST_PREDICT_INTERSECT_COUNT_SAMPLE_RATE)
        self._sampled_locations: bitarray
        if self.index_length < BOOL_FAST_PREDICT_INTERSECT_COUNT_THRESHOLD:
            self._sampled_locations = locations
        else:
            self._sampled_locations = locations[sample_step - 1::sample_step]

        sample_length: int = len(self._sampled_locations)
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
        if loc_list[0].index_length < BOOL_FAST_PREDICT_INTERSECT_COUNT_THRESHOLD:
            estimated_total_nonzero_count: int = sampled_nonzero_count
        else:
            estimated_total_nonzero_count: int = int(
                sampled_nonzero_count / BOOL_FAST_PREDICT_INTERSECT_COUNT_SAMPLE_RATE)
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


class CutPointLocations:

    def __init__(self, cut_point_id: int, lt_loc: IndexLocations, gte_loc: IndexLocations):
        self.cut_point_id: int = cut_point_id
        self.lt_locations: IndexLocations = lt_loc
        self.gte_locations: IndexLocations = gte_loc


class CutPointIndexItem:

    def __init__(self, cut_point_locations: dict[int, CutPointLocations], nan_locations: IndexLocations):
        self.cut_point_locations: dict[int, CutPointLocations] = cut_point_locations
        self.nan_locations: IndexLocations = nan_locations


class Index:

    def __init__(self, data_df: pd.DataFrame, target_column: str | None, target_value: Value | None,
                 column_info: dict[str, ColumnInfo]):
        self.total_count = len(data_df)
        self.data_df = data_df
        self.column_info: dict[str, ColumnInfo] = column_info

        # column -> value -> loc
        self._categorical_index: dict[str, dict[Value, IndexLocations]]
        self._init_categorical_index(data_df)
        # column -> CutPointIndexItem
        self._cut_point_index: dict[str, CutPointIndexItem]
        self._init_cut_point_index(data_df)

        self.target_column: str | None = target_column
        self.target_value: Value | None = target_value
        self.total_target_locations: IndexLocations | None = None
        self.total_target_count: int | None = None
        self.total_target_rate: float | None = None
        if target_column is not None:
            self.total_target_locations = self.get_categorical_locations(target_column, target_value)
            self.total_target_count = self.total_target_locations.count
            self.total_target_rate = self.total_target_locations.count / self.total_count

        filtered_category_columns: list[str] = []
        filtered_continuous_columns: list[str] = []
        for col in data_df.columns:
            col: str
            if col == target_column:
                continue
            col_info: ColumnInfo = self.column_info[col]
            if col_info.binning:
                filtered_continuous_columns.append(col)
            else:
                filtered_category_columns.append(col)
        filtered_columns: list[str] = filtered_category_columns + filtered_continuous_columns
        self.non_target_columns: list[str] = filtered_columns

        self._baseline_coverage_cache: dict[str, float] = {}

    def _init_cut_point_index(self, data_df: pd.DataFrame):
        print("Indexing continuous columns...")
        start = time.time()
        # column -> CutPointIndexItem
        cut_point_index: dict[str, CutPointIndexItem] = {}

        full_loc_bit: bitarray = bitarray(len(data_df))
        full_loc_bit.setall(1)
        full_loc: IndexLocations = IndexLocations(full_loc_bit)

        zero_loc_bit: bitarray = bitarray(len(data_df))
        zero_loc_bit.setall(0)
        zero_loc: IndexLocations = IndexLocations(zero_loc_bit)

        for col_name in data_df.columns:
            if not self.column_info[col_name].binning:
                continue
            print(' - Indexing %s, %d cut points' % (col_name, len(self.column_info[col_name].cut_points)))

            na_loc_bool: np.ndarray = data_df[col_name].isna()
            na_loc_bit: bitarray = bitarray(buffer=np.packbits(na_loc_bool).data)[:len(data_df)]
            na_loc: IndexLocations = IndexLocations(na_loc_bit)

            cut_points: list[float | int] = self.column_info[col_name].cut_points
            cut_point_locations: dict[int, CutPointLocations] = {}
            for cut_point_id in range(0, len(cut_points)):
                cut_point: float | int = cut_points[cut_point_id]
                gte_loc_bool: np.ndarray = data_df[col_name] >= cut_point
                gte_loc_bit: bitarray = bitarray(buffer=np.packbits(gte_loc_bool).data)[:len(data_df)]
                lt_loc_bool: np.ndarray = data_df[col_name] < cut_point
                lt_loc_bit: bitarray = bitarray(buffer=np.packbits(lt_loc_bool).data)[:len(data_df)]
                cut_point_locations[cut_point_id] = (
                    CutPointLocations(cut_point_id, IndexLocations(lt_loc_bit), IndexLocations(gte_loc_bit)))
            cut_point_locations[-1] = CutPointLocations(-1, zero_loc, full_loc)
            cut_point_locations[len(cut_points)] = CutPointLocations(len(cut_points), full_loc, zero_loc)

            cut_point_index[col_name] = CutPointIndexItem(cut_point_locations, na_loc)
        self._cut_point_index = cut_point_index
        print("Index continuous columns cost: %.2f seconds" % (time.time() - start))

    def _init_categorical_index(self, data_df: pd.DataFrame):
        print("Indexing categorical columns...")
        start = time.time()
        categorical_index: dict[str, dict[Value, IndexLocations]] = {}
        for col_name in data_df.columns:
            if self.column_info[col_name].binning:
                continue
            categorical_index[col_name] = {}

        data_array: np.ndarray = data_df.to_numpy(copy=False)
        for col_pos in range(0, len(data_df.columns)):
            col_name: str = data_df.columns[col_pos]
            if self.column_info[col_name].binning:
                continue
            unique_values: pd.Series = pd.Series(data_df[col_name].unique())
            print(' - Indexing %s, unique values: %d' % (col_name, len(unique_values)))
            if len(unique_values) <= 400:
                for val in unique_values:
                    val: Value
                    if val is None or issubclass(type(val), float) and np.isnan(val):
                        na_loc_bool: np.ndarray = data_df[col_name].isna().to_numpy(copy=False)
                        na_loc_bit: bitarray = bitarray(buffer=np.packbits(na_loc_bool).data)
                        na_loc_bit = na_loc_bit[:len(data_df)]
                        if val is None:
                            categorical_index[col_name][None] = IndexLocations(na_loc_bit)
                        else:
                            categorical_index[col_name][np.nan] = IndexLocations(na_loc_bit)
                    else:
                        val_loc_bool: np.ndarray = (data_df[col_name] == val).to_numpy(copy=False)
                        val_loc_bit: bitarray = bitarray(buffer=np.packbits(val_loc_bool).data)
                        val_loc_bit = val_loc_bit[:len(data_df)]
                        categorical_index[col_name][val] = IndexLocations(val_loc_bit)
            else:
                col_index: dict[Value, list[int]] = {}
                for val in data_df[col_name].unique():
                    if issubclass(type(val), float) and np.isnan(val):
                        col_index[np.nan] = []
                    else:
                        col_index[val] = []

                for row_num in range(0, len(data_array)):
                    val: Value = data_array[row_num][col_pos]
                    if val is None:
                        col_index[None].append(row_num)
                    elif issubclass(type(val), float) and np.isnan(val):
                        col_index[np.nan].append(row_num)
                    else:
                        col_index[val].append(row_num)

                for val, row_number_list in col_index.items():
                    val: Value
                    row_number_list: list[int]
                    loc_bit: bitarray = bitarray(len(data_df))
                    loc_bit[row_number_list] = 1
                    categorical_index[col_name][val] = IndexLocations(loc_bit)
        print("Index categorical columns cost: %.2f seconds" % (time.time() - start))
        self._categorical_index = categorical_index

    def get_columns_after(self, column: str | None):
        if column is None:
            return self.non_target_columns
        # TODO 二分查找
        pos = self.non_target_columns.index(column)
        if pos == len(self.non_target_columns) - 1:
            return []
        else:
            return self.non_target_columns[pos + 1:]

    def get_values_by_categorical_column(self, column: str) -> Iterable[Value]:
        return self._categorical_index[column].keys()

    def get_categorical_locations(self, column: str, value: Value) -> IndexLocations:
        return self._categorical_index[column][value]

    def get_cut_point_locations(self, column: str, nan_val: bool, cut_point_id: int | None, gte: bool | None) \
            -> IndexLocations:
        if nan_val:
            return self._cut_point_index[column].nan_locations
        else:
            if gte:
                return self._cut_point_index[column].cut_point_locations[cut_point_id].gte_locations
            else:
                return self._cut_point_index[column].cut_point_locations[cut_point_id].lt_locations

    # def get_column_combination_coverage_baseline(self, column_values: dict[str, Value]) -> float:
    #     if len(column_values) == 0:
    #         return 1
    #     categorical_columns: list[str] = []
    #     continuous_columns: list[str] = []
    #     for col in column_values.keys():
    #         if self.column_info[col].binning:
    #             continuous_columns.append(col)
    #         else:
    #             categorical_columns.append(col)
    #     baseline_coverage: float = 1
    #     if len(categorical_columns) > 0:
    #         cache_key = ','.join(categorical_columns)
    #         categorical_baseline_coverage: float
    #         if cache_key in self._baseline_coverage_cache:
    #             categorical_baseline_coverage = self._baseline_coverage_cache[cache_key]
    #         else:
    #             categorical_unique_combinations: pd.Series = (
    #                 self.data_df[categorical_columns].drop_duplicates().dropna(ignore_index=True))
    #             categorical_baseline_coverage: float = 1 / len(categorical_unique_combinations)
    #             self._baseline_coverage_cache[cache_key] = categorical_baseline_coverage
    #         baseline_coverage *= categorical_baseline_coverage
    #     if len(continuous_columns) > 0:
    #         outer_hypercube_volume: float = 1
    #         inner_hypercube_volume: float = 1
    #         for col in continuous_columns:
    #             val = column_values[col]
    #             if isinstance(val, float) and np.isnan(val):
    #                 raise Exception('nan (%s) is not supported to calculate baseline coverage!' % col)
    #             val_bin: pd.Interval = cast(pd.Interval, val)
    #             q01: float = self.column_info[col].q01
    #             q99: float = self.column_info[col].q99
    #             if val_bin.left < q01 or val_bin.right > q99:
    #                 raise Exception('Column value must be between q01 and q99 (%.2f, %.2f), actual: (%.2f, %.2f)' %
    #                                 (q01, q99, val_bin.left, val_bin.right))
    #             length: float = q99 - q01
    #             left_normalized: float = val_bin.left / length
    #             right_normalized: float = val_bin.right / length
    #             outer_hypercube_volume *= right_normalized
    #             inner_hypercube_volume *= left_normalized
    #         continuous_baseline_coverage: float = outer_hypercube_volume - inner_hypercube_volume
    #         baseline_coverage *= continuous_baseline_coverage
    #     return baseline_coverage

    @staticmethod
    def fast_predict_bool_intersect_count(loc_list: list['IndexLocations']) -> int | None:
        return IndexLocations.fast_predict_bool_intersect_count(loc_list)
