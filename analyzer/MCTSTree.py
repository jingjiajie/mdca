from typing import Iterable

import numpy as np
import pandas as pd

from analyzer.Index import Index, IndexLocations
from analyzer.MCTSTreeNode import MCTSTreeNode
from analyzer.ResultPath import ResultPath, ResultItem
from analyzer.commons import Value, calc_weight

# TODO 动态自适应
SIMULATE_TIMES: int = 10


class ColumnValueWeights:
    def __init__(self, values: list[Value | pd.Interval], weights: np.ndarray):
        self.values: list[Value | pd.Interval] = values
        self.weights: np.ndarray = weights
        self.max_weight: np.float64 = weights.max()
        self.weights_normalized: np.ndarray
        if self.max_weight == 0:
            self.weights_normalized = weights  # all zero
        else:
            self.weights_normalized = weights / weights.sum()




class MCTSTree:

    def __init__(self, data_index: Index, min_error_coverage: float):
        self.data_index: Index = data_index
        self.min_error_coverage: float = min_error_coverage
        self.min_error_count: int = int(data_index.total_error_count * self.min_error_coverage)
        self._root: MCTSTreeNode = MCTSTreeNode(self, None, None, None)
        self._column_value_weights_cache: dict[str, ColumnValueWeights] = {}

    def _get_value_weights_by_column(self, column: str):
        if column in self._column_value_weights_cache:
            return self._column_value_weights_cache[column]

        index: Index = self.data_index
        total_error_loc: IndexLocations = index.get_locations(index.target_column, index.target_value)
        values: list[Value | pd.Interval] = list(index.get_values_by_column(column))
        values = list(filter(lambda v: index.get_locations(column, v).count >= self.min_error_count, values))
        weights: np.ndarray[np.float64] = np.ndarray(shape=len(values), dtype=np.float64)
        i: int = 0
        for val in values:
            val: str
            loc: IndexLocations = index.get_locations(column, val)
            if loc.count == 0:
                weight = 0
            else:
                error_loc: IndexLocations = loc & total_error_loc
                err_coverage: float = error_loc.count / total_error_loc.count
                if err_coverage < self.min_error_coverage:
                    weights[i] = 0
                    i += 1
                    continue
                err_rate: float = error_loc.count / loc.count
                weight: float = calc_weight(1, err_coverage, err_rate, index.total_error_rate)
            weights[i] = weight
            i += 1
        column_value_weights: ColumnValueWeights = ColumnValueWeights(values, weights)
        self._column_value_weights_cache[column] = column_value_weights
        return column_value_weights

    def run(self, times: int):
        print('MCTS start...')
        i: int = 0
        for i in range(0, times):
            print('MCTS round: %d' % i)
            selected_leaf: MCTSTreeNode = self._root.select()
            print("Selected: ", selected_leaf)
            if selected_leaf.children is None:
                print("Expand")
                selected_leaf.expand()
                for child in selected_leaf.children:
                    print('Simulate', child)
                    child.simulate(SIMULATE_TIMES)
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

