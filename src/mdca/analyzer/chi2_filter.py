import numpy as np
import pandas as pd
from bitarray import bitarray
from scipy import stats

from mdca.analyzer.Index import IndexLocations, Index
from mdca.analyzer.ResultPath import ResultItem, ResultPath, CalculatedResult
from mdca.analyzer.commons import Value

CHI2_THRESHOLD: float = 0.05


def _calc_item_loc_to_other_loc(all_items: list[ResultItem]) -> (list[(IndexLocations, IndexLocations)]):
    items_loc_forward: list[IndexLocations] = []
    for i in range(0, len(all_items)):
        if i == 0:
            items_loc_forward.append(all_items[0].locations)
        else:
            prev_loc: IndexLocations = items_loc_forward[i - 1]
            items_loc_forward.append(prev_loc & all_items[i].locations)

    items_loc_backward: list[IndexLocations] = []
    for i in range(0, len(all_items)):
        if i == 0:
            items_loc_backward.append(all_items[len(all_items) - 1].locations)
        else:
            prev_loc: IndexLocations = items_loc_backward[i - 1]
            items_loc_backward.append(prev_loc & all_items[len(all_items) - 1 - i].locations)

    item_loc_to_other_loc: list[(IndexLocations, IndexLocations)] = []
    for i in range(0, len(all_items)):
        item: ResultItem = all_items[i]
        other_items_loc: IndexLocations
        if i == 0:
            other_items_loc = items_loc_backward[len(all_items) - 2]
        elif i == len(all_items) - 1:
            other_items_loc = items_loc_forward[len(all_items) - 2]
        else:
            other_items_loc = items_loc_forward[i - 1] & items_loc_backward[len(all_items) - 1 - (i + 1)]
        item_loc_to_other_loc.append((item.locations, other_items_loc, ))
    return item_loc_to_other_loc


def chi2_filter_distribution(result: CalculatedResult, index: Index) -> CalculatedResult | None:
    # Delete non-cause columns
    if len(result.items) == 1:
        return result
    filtered_items: list[ResultItem] = result.items
    is_inc: bool = result.coverage >= result.baseline_coverage
    while True:
        filter_vector: np.ndarray = np.zeros(len(filtered_items), dtype=bool)
        item_loc_to_other_loc: list[(IndexLocations, IndexLocations)] = _calc_item_loc_to_other_loc(filtered_items)
        for i in range(0, len(filtered_items)):
            item: ResultItem = filtered_items[i]
            item_loc: IndexLocations = item_loc_to_other_loc[i][0]
            other_items_loc: IndexLocations = item_loc_to_other_loc[i][1]
            filtered_result_loc: IndexLocations = item_loc & other_items_loc
            column_values: dict[str, Value | pd.Interval] = {}
            for cur in filtered_items:
                column_values[cur.column] = cur.value
            filtered_result_baseline_coverage: float = index.get_column_combination_coverage_baseline(column_values)
            del column_values[item.column]
            other_items_baseline_coverage: float = index.get_column_combination_coverage_baseline(column_values)
            filtered_result_coverage: float = filtered_result_loc.count / index.total_count
            other_items_coverage: float = other_items_loc.count / index.total_count
            filtered_result_coverage_inc_by_proportion: float = (
                    filtered_result_coverage - filtered_result_baseline_coverage)
            filtered_result_coverage_inc_by_multiple: float = (
                    filtered_result_coverage / filtered_result_baseline_coverage)
            other_items_coverage_inc_by_proportion: float = other_items_coverage - other_items_baseline_coverage
            other_items_coverage_inc_by_multiple: float = other_items_coverage / other_items_baseline_coverage
            if is_inc:
                if (other_items_coverage_inc_by_proportion >= filtered_result_coverage_inc_by_proportion and
                        other_items_coverage_inc_by_multiple >= filtered_result_coverage_inc_by_multiple):
                    continue
            else:
                if (other_items_coverage_inc_by_proportion < filtered_result_coverage_inc_by_proportion and
                        other_items_coverage_inc_by_multiple < filtered_result_coverage_inc_by_multiple):
                    continue
            observed = [
                [(item_loc & other_items_loc).count, (item_loc & ~other_items_loc).count],
                [(~item_loc & other_items_loc).count, (~item_loc & ~other_items_loc).count]
            ]
            if observed[0][1] == 0 or observed[1][0] == 0 or observed[1][1] == 0:
                filter_vector[i] = True
            else:
                chi2, p, dof, expected_freq = stats.chi2_contingency(observed)
                if p <= CHI2_THRESHOLD:
                    filter_vector[i] = True
                else:
                    pass
        filtered_items: list[ResultItem] =\
            [filtered_items[i] for i in range(0, len(filtered_items)) if filter_vector[i]]
        if len(filtered_items) == 1 or np.all(filter_vector == 1):
            break
    if len(filtered_items) == 0:
        return None
    elif len(filtered_items) == len(result.items):
        return result
    loc_total_bit: bitarray = bitarray(filtered_items[0].locations.index_length)
    loc_total_bit.setall(1)
    loc: IndexLocations = IndexLocations(loc_total_bit)
    for item in filtered_items:
        loc &= item.locations
    return ResultPath(filtered_items, loc, 'distribution').calculate(index)


def chi2_filter(result: CalculatedResult, index: Index, search_mode: str) -> CalculatedResult | None:
    # Delete non-cause columns
    if len(result.items) == 1:
        return result
    total_target_loc: IndexLocations = index.total_target_locations
    total_target_rate: float = index.total_target_rate
    is_inc: bool = result.target_rate >= total_target_rate
    filtered_items: list[ResultItem] = result.items
    while True:
        filter_vector: np.ndarray = np.zeros(len(filtered_items), dtype=bool)
        item_loc_to_other_loc: list[(IndexLocations, IndexLocations)] = _calc_item_loc_to_other_loc(filtered_items)
        for i in range(0, len(filtered_items)):
            item_loc: IndexLocations = item_loc_to_other_loc[i][0]
            other_items_loc: IndexLocations = item_loc_to_other_loc[i][1]
            filtered_result_loc: IndexLocations = other_items_loc & item_loc
            other_items_target_loc: IndexLocations = other_items_loc & total_target_loc
            other_items_target_rate: float = other_items_target_loc.count / other_items_loc.count
            filtered_result_target_loc: IndexLocations = filtered_result_loc & total_target_loc
            filtered_result_target_rate: float = filtered_result_target_loc.count / filtered_result_loc.count
            if ((is_inc and other_items_target_rate >= filtered_result_target_rate) or
                    (not is_inc and other_items_target_rate < filtered_result_target_rate)):
                continue
            observed = [
                [(filtered_result_loc & total_target_loc).count, (filtered_result_loc & ~total_target_loc).count],
                [(other_items_loc & ~item_loc & total_target_loc).count, (other_items_loc & ~item_loc & ~total_target_loc).count]
            ]
            if observed[0][1] == 0 or observed[1][0] == 0 or observed[1][1] == 0:
                filter_vector[i] = True
            else:
                chi2, p, dof, expected_freq = stats.chi2_contingency(observed)
                if p <= CHI2_THRESHOLD:
                    filter_vector[i] = True
                else:
                    pass
        filtered_items: list[ResultItem] = [filtered_items[i] for i in range(0, len(filtered_items)) if filter_vector[i]]
        if len(filtered_items) == 1 or np.all(filter_vector == 1):
            break
    if len(filtered_items) == 0:
        return None
    elif len(filtered_items) == len(result.items):
        return result
    loc_total_bit: bitarray = bitarray(filtered_items[0].locations.index_length)
    loc_total_bit.setall(1)
    loc: IndexLocations = IndexLocations(loc_total_bit)
    for item in filtered_items:
        loc &= item.locations
    return ResultPath(filtered_items, loc, search_mode).calculate(index)
