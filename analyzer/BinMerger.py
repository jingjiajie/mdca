from copy import copy
from typing import cast, Iterable

import numpy as np
import pandas as pd

from analyzer.Index import Index
from analyzer.ResultPath import ResultPath, ResultItem, CalculatedResult
from analyzer.commons import Value, calc_weight


class BinMerger:

    def __init__(self, data_index: Index, column_types: dict[str, str], column_binning: dict[str, bool]):
        self.data_index = data_index
        self.column_types: dict[str, str] = column_types
        self.column_binning: dict[str, bool] = column_binning

    def filter(self, results: list[ResultPath]):
        filtered_results: list[ResultPath] = []
        for result_path in results:
            filtered_items: list[ResultItem] = []
            for item in result_path.items:
                if item.locations.sum() == self.data_index.total_count:
                    pass
                else:
                    filtered_items.append(item)
            filtered_results.append(ResultPath(filtered_items))
        return filtered_results

    def merge(self, results: list[ResultPath]):
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
                        if not self.column_binning[col]:
                            if not (item.value == compare_item.value or item.value is compare_item.value):
                                should_merge = False
                                break
                        else:  # Is bin column
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
                    for item in cur_res.items:
                        col: str = item.column
                        if not self.column_binning[col]:
                            merged_res_items.append(item)
                        else:
                            merge_item: ResultItem = merge_res[col]
                            this_bin: pd.Interval = cast(pd.Interval, item.value)
                            merge_bin: pd.Interval = cast(pd.Interval, merge_item.value)
                            left: int = min(this_bin.left, merge_bin.left)
                            right: int = max(this_bin.right, merge_bin.right)
                            new_bin: pd.Interval = pd.Interval(left, right, closed='left')
                            new_loc: pd.Series = item.locations | merge_item.locations
                            new_item = ResultItem(col, new_bin, new_loc)
                            merged_res_items.append(new_item)
                    new_res: ResultPath = ResultPath(merged_res_items)
                    calc_new: CalculatedResult = new_res.calculate(self.data_index)
                    calc_cur: CalculatedResult = cur_res.calculate(self.data_index)
                    calc_compare: CalculatedResult = merge_res.calculate(self.data_index)
                    # TODO and/or?
                    if calc_new.weight >= calc_cur.weight or calc_new.weight >= calc_compare.weight:
                        merge_successful = True
                        group.remove(merge_res)
                        group.append(new_res)
                        break
                if not merge_successful:
                    can_not_merge_list.append(cur_res)
            group = can_not_merge_list
            result_groups[key] = group

        # # Merge bins
        # for key in result_groups.keys():
        #     group: list[ResultPath] = result_groups[key]
        #     can_not_merge_list: list[ResultPath] = []
        #     while len(group) > 0:
        #         cur_res: ResultPath = group.pop()
        #         merge_res: ResultPath | None = None
        #         # Iterate to compare
        #         for i in range(0, len(group)):
        #             compare_res: ResultPath = group[i]
        #             should_merge: bool = True
        #             for item in cur_res.items:
        #                 col: str = item.column
        #                 compare_item: ResultItem = compare_res[col]
        #                 # If it should not merge, break
        #                 if not self.column_binning[col]:
        #                     if not (item.value == compare_item.value or item.value is compare_item.value):
        #                         should_merge = False
        #                         break
        #                 else:  # Is bin column
        #                     this_bin: pd.Interval = cast(pd.Interval, item.value)
        #                     compare_bin: pd.Interval = cast(pd.Interval, compare_item.value)
        #                     if this_bin.right < compare_bin.left or this_bin.left > compare_bin.right:
        #                         should_merge = False
        #                         break
        #             if should_merge:
        #                 merge_res = compare_res
        #         if merge_res is None:
        #             can_not_merge_list.append(cur_res)
        #             continue
        #         merged_res_items: list[ResultItem] = []
        #         for item in cur_res.items:
        #             col: str = item.column
        #             if not self.column_binning[col]:
        #                 merged_res_items.append(item)
        #             else:
        #                 merge_item: ResultItem = merge_res[col]
        #                 this_bin: pd.Interval = cast(pd.Interval, item.value)
        #                 merge_bin: pd.Interval = cast(pd.Interval, merge_item.value)
        #                 left: int = min(this_bin.left, merge_bin.left)
        #                 right: int = max(this_bin.right, merge_bin.right)
        #                 new_bin: pd.Interval = pd.Interval(left, right, closed='left')
        #                 new_loc: pd.Series = item.locations | merge_item.locations
        #                 new_item = ResultItem(col, new_bin, new_loc)
        #                 merged_res_items.append(new_item)
        #         new_res: ResultPath = ResultPath(merged_res_items)
        #         group.remove(merge_res)
        #         group.append(new_res)
        #
        #         # 证明Merge后新weight必然在两个result大小之间：
        #         # new_weight = (cov1 + cov2) * new_rate
        #         #   = (cov1 + cov2) * (err1 + err2)/(cnt1 + cnt2)
        #         # 比较new_rate和rate1比大小，做差
        #         # new_rate - rate1
        #         #   = (err1 + err2)/(cnt1 + cnt2) - err1/cnt1
        #         #   = cnt1(err1+err2)/cnt1(cnt1 + cnt2) - err1(cnt1+cnt2)/cnt1(cnt1+cnt2)
        #         #   = [cnt1(err1+err2) - err1(cnt1+cnt2)] / cnt1(cnt1 + cnt2)
        #         #   = (cnt1*err1 + cnt1*err2 - err1*cnt1 - err1*cnt2)/cnt1(cnt1 + cnt2)
        #         #   = (err2*cnt1 - err1*cnt2)/cnt1(cnt1 + cnt2)
        #         #   = cnt2*(err2/cnt2 - err1/cnt1)/(cnt1+cnt2)
        #         #   = cnt2*(rate2 - rate1) / (cnt1+cnt2)
        #         # 假设rate1 < rate2则>0，即new_rate > rate1，反之new_rate < rate1
        #         # 故new_rate 介于 rate1和rate2中间
        #
        #         # new_weight = (cov1 + cov2) * (new_rate)
        #         # 假设rate1 < rate2，则new_rate > rate1
        #         #   (cov1 + cov2) > cov1
        #         #   (cov1 + cov2) * new_rate > cov1*rate1
        #         #   反之亦然，故new_weight必然在两个result大小之间：
        #
        #     group = can_not_merge_list
        #     result_groups[key] = group

        # Collect results
        final_results: list[ResultPath] = []
        for key in result_groups.keys():
            group: list[ResultPath] = result_groups[key]
            final_results += group
        return final_results

    def expand(self, results: list[ResultPath]):
        final_results: list[ResultPath] = []
        index: Index = self.data_index
        total_loc: pd.Series = np.ones(self.data_index.total_count, dtype=bool)
        total_error_loc: pd.Series = index.get_locations(index.target_column, index.target_value)
        total_error_count: int = total_error_loc.sum()
        for result_path in results:
            expanded_result_items: list[ResultItem] = [m for m in result_path.items]
            if len(result_path.items) == 1 and result_path.items[0].column == 'subGrade_trans':
                pass
            for item_pos in range(0, len(expanded_result_items)):
                result_item: ResultItem = expanded_result_items[item_pos]
                # result_weight: float = ResultPath(expanded_result_items).calculate(index).weight
                tmp_calc = ResultPath(expanded_result_items).calculate(index)
                col: str = result_item.column
                val: Value | pd.Interval = result_item.value
                if not self.column_binning[col] or (type(val) is not pd.Interval):
                    continue
                this_bin: pd.Interval = cast(pd.Interval, val)
                all_bins: list[pd.Interval] = [v for v in index.get_values_by_column(col) if type(v) is pd.Interval]
                # 提前计算缓存
                all_bins_asc = sorted(all_bins, key=lambda interval: interval.left)

                other_items_loc: pd.Series = total_loc
                for i in range(0, len(expanded_result_items)):
                    if expanded_result_items[i].column == col:
                        continue
                    other_items_loc = other_items_loc & expanded_result_items[i].locations

                this_bin_pos: int = 0
                while this_bin_pos < len(all_bins_asc):
                    if all_bins_asc[this_bin_pos] == this_bin:
                        break
                    else:
                        this_bin_pos += 1

                upper_bin_pos: int = this_bin_pos
                lower_bin_pos: int = this_bin_pos
                _merged_bin_loc: pd.Series = index.get_locations(col, this_bin)
                last_weight: float = tmp_calc.weight

                # Merge bin
                for direction in ['up', 'down']:
                    start: int = upper_bin_pos + 1 if direction == 'up' else lower_bin_pos - 1
                    end: int = len(all_bins_asc) if direction == 'up' else -1
                    step: int = 1 if direction == 'up' else -1
                    for bin_pos in range(start, end, step):
                        next_bin: pd.Interval = all_bins_asc[bin_pos]
                        next_loc: pd.Series = index.get_locations(col, next_bin)
                        new_merged_bin_loc = _merged_bin_loc | next_loc
                        new_result_loc: pd.Series = other_items_loc & new_merged_bin_loc
                        new_result_err_loc: pd.Series = new_result_loc & total_error_loc
                        new_result_count: int = new_result_loc.sum()
                        new_error_count: int = new_result_err_loc.sum()
                        new_error_rate: float = new_error_count / new_result_count
                        new_error_coverage: float = new_error_count / total_error_count
                        new_weight: float = calc_weight(len(expanded_result_items), new_error_coverage,
                                                        new_error_rate, index.total_error_rate)
                        if new_weight >= last_weight:
                            # print("###",result_path, ', old:', result_item, ", new:", next_bin,
                            #       ", error_cov: %.2f->%.2f" % (tmp_calc.error_coverage, new_error_coverage),
                            #       ", error_rate: %.2f->%.2f" % (tmp_calc.error_rate, new_error_rate),
                            #       ", weight: %.2f->%.2f" % (result_weight, new_weight))
                            _merged_bin_loc = new_merged_bin_loc
                            last_weight = new_error_rate
                            if direction == 'up':
                                upper_bin_pos = bin_pos
                            else:
                                lower_bin_pos = bin_pos
                        else:
                            break

                # # Remove bin
                # for direction in ['up', 'down']:
                #     start: int = upper_bin_pos if direction == 'down' else lower_bin_pos
                #     end: int = lower_bin_pos if direction == 'down' else upper_bin_pos
                #     step: int = -1 if direction == 'down' else 1
                #     for bin_pos in range(start, end, step):
                #         edge_bin: pd.Interval = all_bins_asc[bin_pos]
                #         edge_bin_loc: pd.Series = index.get_locations(col, edge_bin)
                #         new_merged_bin_loc: pd.Series = _merged_bin_loc & ~edge_bin_loc
                #         new_result_loc: pd.Series = other_items_loc & new_merged_bin_loc
                #         new_result_err_loc: pd.Series = new_result_loc & total_error_loc
                #         new_result_count: int = new_result_loc.sum()
                #         new_error_count: int = new_result_err_loc.sum()
                #         new_error_rate: float = new_error_count / new_result_count
                #         new_error_coverage: float = new_error_count / total_error_count
                #         new_weight: float = calc_weight(new_error_coverage, new_error_rate, index.total_error_rate)
                #         if new_weight >= last_weight:
                #             print("@@@",result_path, ', old:', result_item, ", new:", edge_bin,
                #                   ", error_cov: %.2f->%.2f" % (tmp_calc.error_coverage, new_error_coverage),
                #                   ", error_rate: %.2f->%.2f" % (tmp_calc.error_rate, new_error_rate),
                #                   ", weight: %.2f->%.2f" % (result_weight, new_weight))
                #             _merged_bin_loc = new_merged_bin_loc
                #             last_weight = new_error_rate
                #             if direction == 'down':
                #                 upper_bin_pos = bin_pos
                #             else:
                #                 lower_bin_pos = bin_pos
                #         else:
                #             break

                lower_merge_bin: pd.Interval = all_bins_asc[lower_bin_pos]
                upper_merge_bin: pd.Interval = all_bins_asc[upper_bin_pos]
                merge_bin: pd.Interval
                if lower_merge_bin.left != this_bin.left or upper_merge_bin.right != this_bin.right:
                    merge_bin = pd.Interval(lower_merge_bin.left, upper_merge_bin.right, closed='left')
                else:
                    merge_bin = this_bin
                expanded_result_items[item_pos] = ResultItem(col, merge_bin, _merged_bin_loc)
            final_results.append(ResultPath(expanded_result_items))
        return final_results


