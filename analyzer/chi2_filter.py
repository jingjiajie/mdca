import numpy as np
from bitarray import bitarray
from scipy import stats

from analyzer.Index import IndexLocations
from analyzer.ResultPath import ResultItem, ResultPath

CHI2_THRESHOLD: float = 0.05


def chi2_filter(results: list[ResultPath]) -> list[ResultPath]:

    def _chi2_test_item_pair(item1: ResultItem, item2: ResultItem):
        loc1: IndexLocations = item1.locations
        loc2: IndexLocations = item2.locations
        observed = [
            [(loc1 & loc2).count, (loc1 & ~loc2).count],
            [(~loc1 & loc2).count, (~loc1 & ~loc2).count]
        ]
        if observed[0][1] == 0 or observed[1][0] == 0 or observed[1][1] == 0:
            return None, 0, None, None, None  # TODO
        chi2, p, dof, expected_freq = stats.chi2_contingency(observed)
        return chi2, p, dof, expected_freq, observed

    # Delete non-cause columns
    filtered_results: list[ResultPath] = []
    for cur_result in results:
        if len(cur_result.items) == 1:
            filtered_results.append(cur_result)
            continue
        rel_vector: np.ndarray = np.zeros(len(cur_result.items), dtype=bool)
        for i in range(0, len(cur_result.items)):
            for j in range(i+1, len(cur_result.items)):
                if rel_vector[i] and rel_vector[j]:
                    continue
                item1: ResultItem = cur_result.items[i]
                item2: ResultItem = cur_result.items[j]
                chi2, p, dof, expected_freq, actual_freq = _chi2_test_item_pair(item1, item2)
                if p <= CHI2_THRESHOLD:
                    rel_vector[i] = rel_vector[j] = True
        filtered_items: list[ResultItem] =\
            [cur_result.items[i] for i in range(0, len(cur_result.items)) if rel_vector[i]]
        if len(filtered_items) == 0:
            continue
        loc_total_bit: bitarray = bitarray(filtered_items[0].locations.index_length)
        loc_total_bit.setall(1)
        loc: IndexLocations = IndexLocations(loc_total_bit)
        for item in filtered_items:
            loc &= item.locations
        filtered_results.append(ResultPath(filtered_items, loc))
    return filtered_results
