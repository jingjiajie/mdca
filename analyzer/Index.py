import random
from typing import Iterable

import numpy as np
import pandas as pd


class IndexLocation:

    def __init__(self, start: int, end: int):
        self.start: int = start
        self.end: int = end

    def __str__(self):
        return f"[{self.start}, {self.end}]"


class IndexLocationList:

    def __init__(self, location: IndexLocation | None = None):
        self.locations: list[IndexLocation]  # Ascending ordered
        if location is not None:
            self.locations = [location]
            self.count = location.end - location.start + 1
        else:
            self.locations = []
            self.count = 0

    def append_single_line(self, line_num: int):
        self.count += 1
        if len(self.locations) == 0:
            self.locations.append(IndexLocation(line_num, line_num))
            return
        last_loc = self.locations[len(self.locations) - 1]
        if last_loc.end + 1 == line_num:
            last_loc.end = line_num
        else:
            self.locations.append(IndexLocation(line_num, line_num))

    def intersect(self, target: 'IndexLocationList') -> 'IndexLocationList':
        new_list: IndexLocationList = IndexLocationList()
        if len(self.locations) == 0 or len(target.locations) == 0:
            return new_list
        self_pos: int = 0
        target_pos: int = 0
        while True:
            if self_pos == len(self.locations) or target_pos == len(target.locations):
                break
            self_loc: IndexLocation = self.locations[self_pos]
            target_loc: IndexLocation = target.locations[target_pos]
            if self_loc.end < target_loc.start:
                self_pos += 1
            elif target_loc.end < self_loc.start:
                target_pos += 1
            else:
                max_start: int = self_loc.start
                if target_loc.start > max_start:
                    max_start = target_loc.start

                min_end: int = self_loc.end
                if target_loc.end < min_end:
                    min_end = target_loc.end
                new_list.locations.append(IndexLocation(max_start, min_end))

                if min_end == self_loc.end:
                    self_pos += 1
                else:
                    target_pos += 1

        for loc in new_list.locations:
            new_list.count += loc.end - loc.start
        new_list.count += len(new_list.locations)
        return new_list


class Index:

    def __init__(self, data_df: pd.DataFrame, index_cols: list[str], threshold: int):
        self._index: dict[str, dict[str, IndexLocationList]]
        self._select_vector: dict[str, list[(str, int)]]
        self._init_index(data_df, index_cols, threshold)
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

    def _init_index(self, data_df: pd.DataFrame, index_cols: list[str], threshold: int):
        index: dict[str, dict[str, IndexLocationList]] = {}
        for col in index_cols:
            index[col] = {}
            col_series: pd.Series = data_df[col]
            line_num = 0
            for val in col_series:
                loc_list: IndexLocationList
                if val in index[col]:
                    loc_list = index[col][val]
                else:
                    index[col][val] = loc_list = IndexLocationList()
                loc_list.append_single_line(line_num)
                line_num += 1
        # remove items lower than threshold
        filtered_index: dict[str, dict[str, IndexLocationList]] = {}
        for col in index:
            for val in index[col]:
                loc_list: IndexLocationList = index[col][val]
                if loc_list.count >= threshold:
                    if col not in filtered_index:
                        filtered_index[col] = {}
                    filtered_index[col][val] = loc_list
        self._index = filtered_index

    def _init_select_vector(self):
        self._select_vector = {}  # column -> (value, cumulative_freq)
        for col in self._index.keys():
            val_count_map: dict[str, int] = {}
            index_col: dict[str, IndexLocationList] = self._index[col]
            for key in index_col:
                val = key
                loc_list: IndexLocationList = index_col[key]
                val_count_map[val] = loc_list.count

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

    def get_locations(self, column: str, value: str) -> IndexLocationList:
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

