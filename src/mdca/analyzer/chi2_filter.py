import numpy as np
from bitarray import bitarray
from scipy import stats

from mdca.analyzer.Index import IndexLocations
from mdca.analyzer.ResultPath import ResultItem, ResultPath

CHI2_THRESHOLD: float = 0.05


def _chi2_test_location_pair(loc1: IndexLocations, loc2: IndexLocations):
    observed = [
        [(loc1 & loc2).count, (loc1 & ~loc2).count],
        [(~loc1 & loc2).count, (~loc1 & ~loc2).count]
    ]
    if observed[0][1] == 0 or observed[1][0] == 0 or observed[1][1] == 0:
        return None, 0, None, None, None  # TODO
    chi2, p, dof, expected_freq = stats.chi2_contingency(observed)
    return chi2, p, dof, expected_freq, observed


def chi2_filter(result: ResultPath, search_mode: str) -> ResultPath | None:
    # Delete non-cause columns
    if len(result.items) == 1:
        return result
    filtered_items: list[ResultItem] = result.items
    while True:
        rel_vector: np.ndarray = np.zeros(len(filtered_items), dtype=bool)
        for i in range(0, len(filtered_items)):
            item: ResultItem = result.items[i]
            other_items_loc: IndexLocations | None = None
            for j in range(0, len(filtered_items)):
                if j == i:
                    continue
                if other_items_loc is None:
                    other_items_loc = result.items[j].locations
                else:
                    other_items_loc = other_items_loc & result.items[j].locations
            chi2, p, dof, expected_freq, actual_freq = _chi2_test_location_pair(item.locations, other_items_loc)
            if p <= CHI2_THRESHOLD:
                rel_vector[i] = True
            else:
                pass
        filtered_items: list[ResultItem] =\
            [filtered_items[i] for i in range(0, len(filtered_items)) if rel_vector[i]]
        if len(filtered_items) == 1 or np.all(rel_vector == 1):
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
    return ResultPath(filtered_items, loc, search_mode)
