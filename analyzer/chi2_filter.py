import numpy as np
import pandas as pd
from scipy import stats

from analyzer.Index import  Index
from analyzer.ResultPath import ResultItem, ResultPath


# def get_value_counts(data_df: pd.DataFrame, location_list: IndexLocationList | None, column: str) -> dict[str, int]:
#     series: np.ndarray = data_df[column]
#     value_count: dict[str, int] = {}
#     loc: IndexLocation
#     if location_list is None:
#         location_list = IndexLocationList(IndexLocation(0, len(data_df) - 1))
#     for loc in location_list.locations:
#         start: int = loc.start
#         end: int = loc.end
#         for row in range(start, end + 1):
#             val: str = series[row]
#             if val not in value_count:
#                 value_count[val] = 0
#             value_count[val] += 1
#     return value_count


def chi2_filter(results: list[ResultPath], data_df: pd.DataFrame, index: Index) -> list[ResultPath]:
    # Delete non-cause columns
    filtered_results = []
    for cur_result in results:
        # print("\n@@@@@@@@@@@@@@@@@", cur_result.path())
        cur_result_items: list[ResultItem] = cur_result.items
        no_items_deleted: bool = False
        while not no_items_deleted:
            no_items_deleted = True
            filtered_items: list[ResultItem] = []
            remaining_loc: pd.Series = pd.Series(np.ones(len(data_df), dtype=np.bool))
            for cur_item in cur_result_items:
                item: ResultItem
                for item in cur_result_items:
                    if item.column == cur_item.column:
                        continue
                    loc: pd.Series = index.get_locations(item.column, item.value)
                    remaining_loc = remaining_loc & loc
                cur_item_locations_total: pd.Series = index.get_locations(cur_item.column, cur_item.value)
                cur_item_locations_subset: pd.Series = remaining_loc & cur_item_locations_total
                observed = [[cur_item_locations_subset.sum(), remaining_loc.sum()],
                            [cur_item_locations_total.sum(), len(data_df)]]
                chi2, p, dof, expected_freq = stats.chi2_contingency(observed)
                expected_cur_item_freq_subset: np.float64 = expected_freq[0][0]
                # print(cur_item, "\t","P: %.2f" % p,"SUBSET: ", value_count_subset, "\t/ TOTAL:", value_count_total)
                if p <= 0.05 and cur_item_locations_subset.sum() > expected_cur_item_freq_subset:
                    filtered_items.append(cur_item)
                else:  # item deleted
                    no_items_deleted = False
            cur_result_items = filtered_items
        filtered_results.append(ResultPath(cur_result_items))
    return filtered_results
