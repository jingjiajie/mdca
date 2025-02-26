import time

import numpy as np
import pandas as pd
from bitarray import bitarray

from analyzer.Index import Index, IndexLocations, IndexLocationType
from analyzer.MCTSTreeNode import MCTSTreeNode
from analyzer.ResultPath import ResultPath, ResultItem
from analyzer.commons import Value


class MCTSTree:

    def __init__(self, data_index: Index, column_types: dict[str, str], min_error_coverage: float):
        self.data_index: Index = data_index
        self.column_types: dict[str, str] = column_types
        self._root: MCTSTreeNode

        self.min_error_coverage: float = min_error_coverage
        self.min_error_count: int = int(data_index.total_error_count * self.min_error_coverage)
        self._column_values_satisfy_min_error_coverage: dict[str, dict[Value | pd.Interval, IndexLocations]] = {}
        for col in data_index.get_columns_after(None):
            self._column_values_satisfy_min_error_coverage[col] = {}
            for val in data_index.get_values_by_column(col):
                loc: IndexLocations = data_index.get_locations(col, val)
                if loc.count < self.min_error_count:
                    continue
                self._column_values_satisfy_min_error_coverage[col][val] = loc

    def _reset(self):
        root_loc: bitarray = bitarray(self.data_index.total_count)
        root_loc.setall(1)
        self._root = MCTSTreeNode(self, None, None, None, IndexLocations(root_loc))

    def _get_values_satisfy_min_error_coverage_by_column(self, column: str) \
            -> dict[Value | pd.Interval, IndexLocations]:
        return self._column_values_satisfy_min_error_coverage[column]

    def run(self, times: int):
        print('MCTS start...')
        self._reset()
        i: int = 0
        for i in range(0, times):
            if i != 0 and (i+1) % 1000 == 0:
                print('MCTS round: %d' % (i+1))
            selected_leaf: MCTSTreeNode = self._root.select()
            if selected_leaf.children is None:
                selected_leaf.expand()
                assert selected_leaf.children is not None
                for child in selected_leaf.children:
                    child.simulate()
                    child.back_propagate()
            elif selected_leaf.is_root:
                break
            else:
                raise Exception('Unexpected error: MCTS selection of ' + str(selected_leaf))
        print("MCTS ended, rounds: %d" % i)
        results: list[ResultPath] = self._select_results()
        return results

    # TODO max_results动态适应
    def _select_results(self, max_results: int = 1000) -> list[ResultPath]:
        results: list[ResultPath] = []
        for i in range(0, max_results):
            cur: MCTSTreeNode = self._root
            while cur.children is not None and len(cur.children) > 0:
                max_q_child: MCTSTreeNode = cur.children[0]
                for child in cur.children:
                    if child.max_weight > max_q_child.max_weight:
                        max_q_child = child
                if max_q_child.max_weight < cur.max_weight:
                    break
                else:
                    cur = max_q_child
            if cur.is_root:
                break
            selected_node: MCTSTreeNode = cur
            selected_node.pick()
            result_items: list[ResultItem] = []
            cur = selected_node
            while cur.parent is not None:
                result_items.append(
                    ResultItem(cur.column, self.column_types[cur.column], cur.value,
                               self.data_index.get_locations(cur.column, cur.value)))
                cur = cur.parent
            result_items.reverse()
            result_path: ResultPath = ResultPath(result_items, selected_node.locations)
            results.append(result_path)
        filtered_results: list[ResultPath] = self._filter_out_subset_result(results)
        return filtered_results

    @staticmethod
    def _filter_out_subset_result(results: list[ResultPath]) -> list[ResultPath]:
        ordered_res: list[ResultPath] = sorted(results, key=lambda r: len(r.items), reverse=True)
        is_subset_idx_set: set[int] = set()
        for i in range(len(ordered_res)):
            cur_res: ResultPath = ordered_res[i]
            for j in range(i+1, len(ordered_res)):
                compare_res: ResultPath = ordered_res[j]
                if len(cur_res.items) == len(compare_res.items):
                    continue
                is_subset: bool = True
                for compare_item in compare_res.items:
                    cur_item: ResultItem | None = cur_res[compare_item.column]
                    if cur_item is None:
                        is_subset = False
                        break
                    elif cur_item.value != compare_item.value:
                        is_subset = False
                        break
                if is_subset:
                    is_subset_idx_set.add(j)
        filtered_res: list[ResultPath] = []
        for i in range(len(ordered_res)):
            if i in is_subset_idx_set:
                continue
            filtered_res.append(ordered_res[i])
        return filtered_res


