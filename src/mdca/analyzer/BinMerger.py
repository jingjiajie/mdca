import time
from typing import cast

import numpy as np
import pandas as pd
from bitarray import bitarray

from mdca.analyzer.Index import Index, IndexLocations
from mdca.analyzer.ResultPath import ResultPath, ResultItem, CalculatedResult
from mdca.analyzer.commons import Value, calc_weight_fairness, calc_weight_distribution, ColumnInfo, calc_weight_error


class BinMerger:

    def __init__(self, data_index: Index, column_info: dict[str, ColumnInfo], search_mode: str):
        self.data_index = data_index
        self.column_info: dict[str, ColumnInfo] = column_info
        self.search_mode: str = search_mode

    def filter(self, results: list[ResultPath]):
        print("Filtering bins...")
        start_time: float = time.time()
        filtered_results: list[ResultPath] = []
        for result_path in results:
            filtered_items: list[ResultItem] = []
            for item in result_path.items:
                col_info: ColumnInfo = self.column_info[item.column]
                if isinstance(item.value, pd.Interval):
                    val_bin: pd.Interval = cast(pd.Interval, item.value)
                    if val_bin.left == col_info.q01 and val_bin.right == col_info.q99:
                        continue
                filtered_items.append(item)
            if len(filtered_items) == len(result_path.items):
                filtered_results.append(result_path)
            elif len(filtered_items) == 0:
                continue
            else:
                total_loc_bit: bitarray = bitarray(filtered_items[0].locations.index_length)
                total_loc_bit.setall(1)
                loc: IndexLocations = IndexLocations(total_loc_bit)
                for item in filtered_items:
                    loc &= item.locations
                filtered_results.append(ResultPath(filtered_items, loc, self.search_mode))
        print("Filter cost: %.2f seconds" % (time.time() - start_time))
        return filtered_results

    def merge(self, results: list[ResultPath]):
        print("Merging bins...")
        start_time: float = time.time()
        result_groups: dict[str, list[ResultPath]] = {}
        # Grouping by column
        for result_path in results:
            columns: list[str] = [item.column for item in result_path.items]
            key: str = ",".join(columns)
            if key not in result_groups:
                result_groups[key] = []
            result_groups[key].append(result_path)

        # Merge
        for key in result_groups.keys():
            group: list[ResultPath] = result_groups[key]
            can_not_merge_list: list[ResultPath] = []
            while len(group) > 0:
                cur_res: ResultPath = group.pop()
                try_list: list[ResultPath] = []
                # Iterate to compare
                for i in range(0, len(group)):
                    compare_res: ResultPath = group[i]
                    should_merge: bool = True
                    for item in cur_res.items:
                        col: str = item.column
                        compare_item: ResultItem = compare_res[col]
                        # If it should not merge, break
                        if not self.column_info[col].binning:
                            if not (item.value == compare_item.value or item.value is compare_item.value):
                                should_merge = False
                                break
                        else:  # Is bin column
                            if (issubclass(type(item.value), float)  # To handle nan
                                    or issubclass(type(compare_item.value), float)):
                                if item.value is compare_item:
                                    continue
                                else:
                                    should_merge = False
                                    break
                            this_bin: pd.Interval = cast(pd.Interval, item.value)
                            compare_bin: pd.Interval = cast(pd.Interval, compare_item.value)
                            if this_bin.right < compare_bin.left or this_bin.left > compare_bin.right:
                                should_merge = False
                                break
                    if should_merge:
                        try_list.append(compare_res)
                if len(try_list) == 0:
                    can_not_merge_list.append(cur_res)
                    continue

                # Try to merge
                merge_successful: bool = False
                for merge_res in try_list:
                    merged_res_items: list[ResultItem] = []
                    all_overlapped: bool = True
                    for item in cur_res.items:
                        col: str = item.column
                        if not self.column_info[col].binning:
                            merged_res_items.append(item)
                        else:
                            merge_item: ResultItem = merge_res[col]
                            new_bin: pd.Interval | float
                            if (item.value is merge_item.value  # Handle nan
                                    and issubclass(type(item.value), float) and np.isnan(item.value)):
                                new_bin = item.value
                            else:
                                this_bin: pd.Interval = cast(pd.Interval, item.value)
                                merge_bin: pd.Interval = cast(pd.Interval, merge_item.value)
                                left: int = min(this_bin.left, merge_bin.left)
                                right: int = max(this_bin.right, merge_bin.right)
                                overlapped: bool = max(this_bin.left, merge_bin.left) <= min(this_bin.right,
                                                                                             merge_bin.right)
                                if not overlapped:
                                    all_overlapped = False
                                new_bin = pd.Interval(left, right, closed='left')
                            new_loc: IndexLocations = item.locations | merge_item.locations
                            new_item = ResultItem(col, self.column_info[col].column_type, new_bin, new_loc)
                            merged_res_items.append(new_item)
                    total_loc_bit: bitarray = bitarray(merged_res_items[0].locations.index_length)
                    total_loc_bit.setall(1)
                    new_res_loc: IndexLocations = IndexLocations(total_loc_bit)
                    for item in merged_res_items:
                        new_res_loc &= item.locations
                    new_res: ResultPath = ResultPath(merged_res_items, new_res_loc, self.search_mode)
                    calc_new: CalculatedResult = new_res.calculate(self.data_index)
                    calc_cur: CalculatedResult = cur_res.calculate(self.data_index)
                    calc_compare: CalculatedResult = merge_res.calculate(self.data_index)
                    if calc_new.weight >= min(calc_cur.weight, calc_compare.weight):
                        # TODO 合并策略
                        # if calc_new.target_rate >= min(calc_cur.target_rate, calc_compare.target_rate):
                        # if all_overlapped:
                        # if True:
                        merge_successful = True
                        group.remove(merge_res)
                        group.append(new_res)
                        break
                if not merge_successful:
                    can_not_merge_list.append(cur_res)
            group = can_not_merge_list
            result_groups[key] = group

        # Collect results
        final_results: list[ResultPath] = []
        for key in result_groups.keys():
            group: list[ResultPath] = result_groups[key]
            final_results += group
        print("Merge cost: %.2f seconds" % (time.time() - start_time))
        return final_results

    def expand(self, results: list[ResultPath]):
        print("Expanding bins...")
        start_time: float = time.time()
        final_results: list[ResultPath] = []
        index: Index = self.data_index
        for result_path in results:
            expanded_result_items: list[ResultItem] = [m for m in result_path.items]
            expanded_result_loc: IndexLocations = result_path.locations
            for item_pos in range(0, len(expanded_result_items)):
                result_item: ResultItem = expanded_result_items[item_pos]
                result_calc: CalculatedResult = \
                    ResultPath(expanded_result_items, expanded_result_loc, self.search_mode).calculate(index)
                col: str = result_item.column
                col_info: ColumnInfo = self.column_info[col]
                val: Value | pd.Interval = result_item.value
                if not self.column_info[col].binning or (type(val) is not pd.Interval):
                    continue
                this_bin: pd.Interval = cast(pd.Interval, val)
                all_bins: list[pd.Interval] = \
                    [v for v in index.get_values_by_column(col) if type(v) is pd.Interval]
                all_bins_asc = sorted(all_bins, key=lambda interval: interval.left)

                total_loc_bit: bitarray = bitarray(expanded_result_items[0].locations.index_length)
                total_loc_bit.setall(1)
                other_items_loc: IndexLocations = IndexLocations(total_loc_bit)
                for i in range(0, len(expanded_result_items)):
                    if expanded_result_items[i].column == col:
                        continue
                    other_items_loc &= expanded_result_items[i].locations

                this_bin_pos: int = 0
                while this_bin_pos < len(all_bins_asc):
                    if all_bins_asc[this_bin_pos] == this_bin:
                        break
                    else:
                        this_bin_pos += 1

                upper_bin_pos: int = this_bin_pos
                lower_bin_pos: int = this_bin_pos
                _merged_bin_loc: IndexLocations = index.get_locations(col, this_bin)
                last_weight: float = result_calc.weight

                # Merge bin
                for direction in ['up', 'down']:
                    start: int = upper_bin_pos + 1 if direction == 'up' else lower_bin_pos - 1
                    end: int = len(all_bins_asc) if direction == 'up' else -1
                    step: int = 1 if direction == 'up' else -1
                    for bin_pos in range(start, end, step):
                        next_bin: pd.Interval = all_bins_asc[bin_pos]
                        if next_bin.left < col_info.q01 or next_bin.right > col_info.q99:
                            break
                        next_loc: IndexLocations = index.get_locations(col, next_bin)
                        new_merged_bin_loc = _merged_bin_loc | next_loc
                        new_result_loc: IndexLocations = other_items_loc & new_merged_bin_loc
                        new_result_count: int = new_result_loc.count
                        new_weight: float = -1
                        if self.search_mode == 'fairness':
                            total_target_loc: IndexLocations = index.get_locations(index.target_column,
                                                                                   index.target_value)
                            new_result_target_loc: IndexLocations = new_result_loc & total_target_loc
                            new_target_count: int = new_result_target_loc.count
                            new_target_rate: float = new_target_count / new_result_count
                            new_coverage: float = new_result_loc.count / index.total_count
                            new_weight = calc_weight_fairness(len(expanded_result_items), new_coverage,
                                                              new_target_rate, index.total_target_rate)
                        elif self.search_mode == 'error':
                            total_error_loc: IndexLocations = index.get_locations(index.target_column,
                                                                                   index.target_value)
                            total_error_count: int = total_error_loc.count
                            new_result_error_loc: IndexLocations = new_result_loc & total_error_loc
                            new_error_count: int = new_result_error_loc.count
                            new_error_rate: float = new_error_count / new_result_count
                            new_error_coverage: float = new_error_count / total_error_count
                            new_weight = calc_weight_error(len(expanded_result_items), new_error_coverage,
                                                           new_error_rate, index.total_target_rate)
                        elif self.search_mode == 'distribution':
                            new_coverage: float = new_result_count / index.total_count
                            expanded_column_values: dict[str, Value | pd.Interval] = {}
                            for item in expanded_result_items:
                                expanded_column_values[item.column] = item.value
                            baseline_coverage: float = (
                                index.get_column_combination_coverage_baseline(expanded_column_values))
                            new_weight = calc_weight_distribution(
                                len(expanded_result_items), new_coverage, baseline_coverage)
                        if new_weight >= last_weight:
                            _merged_bin_loc = new_merged_bin_loc
                            last_weight = new_weight
                            if direction == 'up':
                                upper_bin_pos = bin_pos
                            else:
                                lower_bin_pos = bin_pos
                        else:
                            break

                lower_merge_bin: pd.Interval = all_bins_asc[lower_bin_pos]
                upper_merge_bin: pd.Interval = all_bins_asc[upper_bin_pos]
                merge_bin: pd.Interval
                if lower_merge_bin.left != this_bin.left or upper_merge_bin.right != this_bin.right:
                    merge_bin = pd.Interval(lower_merge_bin.left, upper_merge_bin.right, closed='left')
                else:
                    merge_bin = this_bin
                expanded_result_items[item_pos] = (
                    ResultItem(col, self.column_info[col].column_type, merge_bin, _merged_bin_loc))
                expanded_result_loc = other_items_loc & _merged_bin_loc
            final_results.append(ResultPath(expanded_result_items, expanded_result_loc, self.search_mode))
        print("Expand cost: %.2f seconds" % (time.time() - start_time))
        return final_results
