import numpy as np
import pandas as pd
from scipy import stats

from analyzer.Index import  Index
from analyzer.ResultPath import ResultItem, ResultPath


def chi2_filter(results: list[ResultPath], target_column: str, index: Index) -> list[ResultPath]:
    full_location: pd.Series = pd.Series(np.ones(index.total_count, dtype=np.bool))
    # Delete non-cause columns
    filtered_results = []
    for cur_result in results:
        cur_result_items: list[ResultItem] = cur_result.items
        no_items_deleted: bool = False
        target_column_deleted: bool = False
        while not no_items_deleted and not target_column_deleted:
            no_items_deleted = True
            filtered_items: list[ResultItem] = []
            for cur_item in cur_result_items:
                remaining_loc: pd.Series = full_location
                item: ResultItem
                for item in cur_result_items:
                    if item.column == cur_item.column:
                        continue
                    loc: pd.Series = index.get_locations(item.column, item.value)
                    remaining_loc = remaining_loc & loc
                cur_item_locations_total: pd.Series = index.get_locations(cur_item.column, cur_item.value)
                cur_item_locations_subset: pd.Series = remaining_loc & cur_item_locations_total
                observed = [[cur_item_locations_subset.sum(), remaining_loc.sum()],
                            [cur_item_locations_total.sum(), index.total_count]]
                chi2, p, dof, expected_freq = stats.chi2_contingency(observed)
                expected_cur_item_freq_subset: np.float64 = expected_freq[0][0]
                if p <= 0.02 and cur_item_locations_subset.sum() > expected_cur_item_freq_subset:
                    filtered_items.append(cur_item)
                else:  # item deleted
                    no_items_deleted = False
                    if cur_item.column == target_column:
                        target_column_deleted = True
                        break
                if target_column_deleted:
                    break
            cur_result_items = filtered_items
        if not target_column_deleted:
            filtered_results.append(ResultPath(cur_result_items))
    return filtered_results
