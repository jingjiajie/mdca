import numpy as np
import pandas as pd
from scipy import stats

from analyzer.Index import  Index
from analyzer.ResultPath import ResultItem, ResultPath
from analyzer.types import Value

CHI2_THRESHOLD: float = 0.01


def chi2_filter(results: list[ResultPath], target_column: str, target_value: Value, index: Index) -> list[ResultPath]:
    full_location: pd.Series = pd.Series(np.ones(index.total_count, dtype=np.bool))
    target_item: ResultItem = ResultItem(target_column, target_value)

    def _chi2_test_item_to_result(cur_result_items: list[ResultItem], cur_item: ResultItem):
        remaining_loc: pd.Series = full_location
        item: ResultItem
        for item in cur_result_items:
            if item.column == cur_item.column:
                continue
            loc: pd.Series = index.get_locations(item.column, item.value)
            remaining_loc = remaining_loc & loc
        cur_item_locations_total: pd.Series = index.get_locations(cur_item.column, cur_item.value)
        cur_item_locations_subset: pd.Series = remaining_loc & cur_item_locations_total
        cur_item_subset_count = cur_item_locations_subset.sum()
        cur_item_total_count = cur_item_locations_total.sum()
        observed = [[cur_item_subset_count, remaining_loc.sum() - cur_item_subset_count],
                    [cur_item_total_count, index.total_count - cur_item_total_count]]
        chi2, p, dof, expected_freq = stats.chi2_contingency(observed)
        return chi2, p, dof, expected_freq, observed

    def _chi2_test_item_pair(item1: ResultItem, item2: ResultItem):
        loc1 = index.get_locations(item1.column, item1.value)
        loc2 = index.get_locations(item2.column, item2.value)
        observed = [
            [(loc1 & loc2).sum(), (loc1 & ~loc2).sum()],
            [(~loc1 & loc2).sum(), (~loc1 & ~loc2).sum()]
        ]
        chi2, p, dof, expected_freq = stats.chi2_contingency(observed)
        return chi2, p, dof, expected_freq, observed

    # Delete non-cause columns
    filtered_results = []
    split_results = []
    for cur_result in results:
        chi2, p, dof, expected_freq, actual_freq = _chi2_test_item_to_result(cur_result.items, target_item)
        # if target column isn't related, delete the result.
        if p > CHI2_THRESHOLD:
            continue
        cur_result_items: list[ResultItem] = cur_result.items
        no_items_deleted: bool = False
        if len(cur_result_items) > 1:
            while not no_items_deleted:
                no_items_deleted = True
                filtered_items: list[ResultItem] = []
                for cur_item in cur_result_items:
                    chi2, p, dof, expected_freq, actual_freq = _chi2_test_item_to_result(cur_result_items, cur_item)
                    # expected_cur_item_freq_subset: np.float64 = expected_freq[0][0]
                    if p <= CHI2_THRESHOLD:  # and cur_item_locations_subset.sum() > expected_cur_item_freq_subset:
                        filtered_items.append(cur_item)
                    else:  # item deleted
                        no_items_deleted = False
                        chi2, p, dof, expected_freq, actual_freq = _chi2_test_item_pair(target_item, cur_item)
                        if p <= CHI2_THRESHOLD:
                            split_results.append(ResultPath([cur_item]))
                cur_result_items = filtered_items
        for i in range(0, len(cur_result_items)):
            for j in range(i+1, len(cur_result_items)):
                item1: ResultItem = cur_result_items[i]
                item2: ResultItem = cur_result_items[j]
                if item1.column == target_column or item2.column == target_column:
                    continue
                loc1 = index.get_locations(item1.column, item1.value)
                loc2 = index.get_locations(item2.column, item2.value)
                observed = [
                    [(loc1 & loc2).sum(), (loc1 & ~loc2).sum()],
                    [(~loc1 & loc2).sum(), (~loc1 & ~loc2).sum()]
                ]
                chi2, p, dof, expected_freq = stats.chi2_contingency(observed)
                expected_joint_freq: int = expected_freq[0][0]
                print("TEST: ", item1, item2, p, (loc1 & loc2).sum() > expected_joint_freq)
        filtered_results.append(ResultPath(cur_result_items))
    return filtered_results + split_results
