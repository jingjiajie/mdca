import time
from typing import Iterable

import numpy as np
import pandas as pd

from analyzer.Index import Index, IndexLocations, IndexLocationType
from analyzer.MCTSTreeNode import MCTSTreeNode
from analyzer.ResultPath import ResultPath, ResultItem
from analyzer.commons import Value, calc_weight

# TODO 动态自适应
SIMULATE_TIMES: int = 10


class MCTSTree:

    def __init__(self, data_index: Index, min_error_coverage: float):
        self.data_index: Index = data_index
        self.min_error_coverage: float = min_error_coverage
        self.min_error_count: int = int(data_index.total_error_count * self.min_error_coverage)
        self._root: MCTSTreeNode = MCTSTreeNode(self, None, None, None,
                                                IndexLocations(data_index, np.ones(data_index.total_count, dtype=bool)))

        self._column_values_satisfy_min_error_coverage: dict[str, dict[Value | pd.Interval, (IndexLocations, IndexLocations)]] = {}
        for col in data_index.get_columns_after(None):
            self._column_values_satisfy_min_error_coverage[col] = {}
            for val in data_index.get_values_by_column(col):
                loc: IndexLocations = data_index.get_locations(col, val)
                if loc.count < self.min_error_count:
                    continue
                err_loc: IndexLocations = loc & data_index.total_error_locations
                if err_loc.count < self.min_error_count:
                    continue
                err_loc.cache(IndexLocationType.BOOL)
                self._column_values_satisfy_min_error_coverage[col][val] = (loc, err_loc)

    def _get_values_satisfy_min_error_coverage_by_column(self, column: str) \
            -> dict[Value | pd.Interval, (IndexLocations, IndexLocations)]:
        return self._column_values_satisfy_min_error_coverage[column]

    def run(self, times: int):
        print('MCTS start...')
        i: int = 0
        for i in range(0, times):
            print('\n--- MCTS round: %d' % i)
            selected_leaf: MCTSTreeNode = self._root.select()
            print("Selected: ", selected_leaf.path())
            if selected_leaf.children is None:
                start = time.time()
                selected_leaf.expand()
                print("Expand cost [%dms], children: %d" % ((time.time() - start)*1000, len(selected_leaf.children)))
                for child in selected_leaf.children:
                    child.simulate()
                    child.back_propagate()
            elif selected_leaf.is_root:
                break
            else:
                raise Exception('Unexpected error: MCTS selection of ' + str(selected_leaf))
        print("MCTS ended, rounds: %d" % i)
        results: list[ResultPath] = self._choose_results()
        return results

    def _choose_results(self, max_results: int = 1000) -> list[ResultPath]:
        results: list[ResultPath] = []
        for i in range(0, max_results):
            cur: MCTSTreeNode = self._root
            while cur.children is not None and len(cur.children) > 0:
                max_q_child: MCTSTreeNode = cur.children[0]
                for child in cur.children:
                    if child.q_value > max_q_child.q_value:
                        max_q_child = child
                if max_q_child.q_value < cur.q_value:
                    break
                else:
                    cur = max_q_child
            if cur.is_root:
                break
            cur.pick()
            result_items: list[ResultItem] = []
            while cur.parent is not None:
                result_items.append(ResultItem(cur.column, cur.value, self.data_index.get_locations(cur.column, cur.value)))
                cur = cur.parent
            result_items.reverse()
            result_path: ResultPath = ResultPath(result_items)
            results.append(result_path)
        filtered_results: list[ResultPath] = self._filter_suffix_result(results)
        return filtered_results

    def _filter_suffix_result(self, results: list[ResultPath]) -> list[ResultPath]:
        filtered_results: list[ResultPath] = []
        for i in range(0, len(results)):
            res1: ResultPath = results[i]
            should_filter_out: bool = False
            for j in range(0, len(results)):
                if i == j:
                    continue
                res2: ResultPath = results[j]
                if len(res1.items) >= len(res2.items):
                    continue
                is_suffix: bool = True
                # Check suffix
                for k in range(0, len(res1.items)):
                    item1: ResultItem = res1.items[-k-1]
                    item2: ResultItem = res2.items[-k-1]
                    if item1 != item2:
                        is_suffix = False
                        break
                if not is_suffix:
                    continue
                if res1.locations.count != res2.locations.count:
                    continue
                should_filter_out = True
            if not should_filter_out:
                filtered_results.append(res1)
        return filtered_results

