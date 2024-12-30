import random
import pandas as pd


class IndexLocation:
    start: int
    end: int

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def __str__(self):
        return f"[{self.start}, {self.end}]"


class IndexItem:

    def __init__(self):
        self.locations: list[IndexLocation] = []  # Ascending ordered
        self.count = 0

    def add_location(self, line_num: int):
        self.count += 1
        if len(self.locations) == 0:
            self.locations.append(IndexLocation(line_num, line_num))
            return
        last_loc = self.locations[len(self.locations) - 1]
        if last_loc.end + 1 == line_num:
            last_loc.end = line_num
        else:
            self.locations.append(IndexLocation(line_num, line_num))


class Index:

    def __init__(self, data_df: pd.DataFrame, index_cols: list[str]):
        self._init_index(data_df, index_cols)
        self._init_select_vector()
        self.columns: list[str] = data_df.columns

    def _init_index(self, data_df: pd.DataFrame, index_cols: list[str]) -> dict[str, dict[str, IndexItem]]:
        self._index: dict[str, dict[str, IndexItem]] = {}  # column -> value -> IndexItem

        index: dict[str, dict[str, IndexItem]] = self._index
        for col in index_cols:
            index[col] = {}
            col_series: pd.Series = data_df[col]
            line_num = 0
            for item in col_series:
                idx_item: IndexItem
                if item in index[col]:
                    idx_item = index[col][item]
                else:
                    index[col][item] = idx_item = IndexItem()
                idx_item.add_location(line_num)
                line_num += 1
        return index

    def _init_select_vector(self):
        self._select_vector: dict[str, list[(str, int)]] = {}

        for col in self._index.keys():
            val_count_map: dict[str, int] = {}
            index_col: dict[str, IndexItem] = self._index[col]
            for key in index_col:
                val = key
                item: IndexItem = index_col[key]
                val_count_map[val] = item.count

            total = 0
            cumulate_list: list[(str, int)] = []
            for key in val_count_map:
                val = key
                count = val_count_map[val]
                total += count
                cumulate_list.append((val, total))
            self._select_vector[col] = cumulate_list

    def get_columns_after(self, column: str):
        # TODO 二分查找
        pos = self.columns.index(column)
        return self.columns[pos:]

    def random_select_by_freq(self, col: str):
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
