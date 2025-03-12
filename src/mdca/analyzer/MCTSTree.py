import time
from typing import cast

import pandas as pd
from bitarray import bitarray

from mdca.analyzer.Index import Index, IndexLocations
from mdca.analyzer.MCTSTreeNode import MCTSTreeNode, TreeNodeState
from mdca.analyzer.ResultPath import ResultPath
from mdca.analyzer.commons import Value, ColumnInfo


class MCTSTree:

    def __init__(self, data_index: Index, column_info: dict[str, ColumnInfo], target_column: str,
                 target_value: Value | None, search_mode: str,
                 min_coverage: float | None, min_target_coverage: float | None):
        if min_coverage is None and min_target_coverage is None:
            raise Exception('At least one of min_coverage or min_target_coverage must be specified!')
        self.data_index: Index = data_index
        self.column_info: dict[str, ColumnInfo] = column_info
        self.target_column: str = target_column
        self.target_value: Value | None = target_value
        self.search_mode: str = search_mode
        self._root: MCTSTreeNode

        self.min_coverage: float | None = min_coverage
        self.min_target_coverage: float | None = min_target_coverage
        self.min_count: int = 0
        self.min_target_count: int = 0
        if min_coverage is not None:
            self.min_count = int(data_index.total_count * self.min_coverage)
        if min_target_coverage is not None:
            self.min_target_count = int(data_index.total_target_count * min_target_coverage)

        self._column_values_candidate: dict[str, dict[Value | pd.Interval, IndexLocations]] = {}
        for col in data_index.get_columns_after(None):
            self._column_values_candidate[col] = {}
            col_info: ColumnInfo = column_info[col]
            for val in data_index.get_values_by_column(col):
                if col_info.binning and not isinstance(val, float):  # isinstance(val, float) to handle nan
                    val_bin: pd.Interval = cast(pd.Interval, val)
                    if val_bin.left < col_info.q01 or val_bin.right > col_info.q99:
                        continue
                loc: IndexLocations = data_index.get_locations(col, val)
                if loc.count < self.min_count:
                    continue
                if data_index.target_column is not None:
                    target_loc: IndexLocations = loc & data_index.total_target_locations
                    if target_loc.count < self.min_target_count:
                        continue
                self._column_values_candidate[col][val] = loc

    def _reset(self):
        root_loc: bitarray = bitarray(self.data_index.total_count)
        root_loc.setall(1)
        self._root = MCTSTreeNode(self, None, None, None, IndexLocations(root_loc))

    def _get_candidate_values_by_column(self, column: str) -> dict[Value | pd.Interval, IndexLocations]:
        return self._column_values_candidate[column]

    def run(self, times: int):
        print('MCTS start...')
        start_time: float = time.time()
        self._reset()
        i: int = 0
        for i in range(0, times):
            if i != 0 and (i+1) % 1000 == 0:
                print(' - MCTS round: %d' % (i+1))
            selected_leaf: MCTSTreeNode = self._root.select()
            if selected_leaf.children is None:
                selected_leaf.expand()
                assert selected_leaf.children is not None
                for child in selected_leaf.children.values():
                    child.simulate()
                    child.back_propagate()
            elif selected_leaf.is_root:
                break
            else:
                raise Exception('Unexpected error: MCTS selection of ' + str(selected_leaf))
        print("MCTS ended, rounds: %d, cost: %.2f seconds" % (i, time.time() - start_time))

    def next_result(self) -> ResultPath | None:
        if self._root.state == TreeNodeState.FULL_PICKED_STATE:
            return None
        cur: MCTSTreeNode = self._root
        while cur.children is not None and len(cur.children) > 0:
            max_weight_child: MCTSTreeNode | None = None
            for child in cur.children.values():
                if child.state != TreeNodeState.FULL_PICKED_STATE:
                    if max_weight_child is None or child.max_weight > max_weight_child.max_weight:
                        max_weight_child = child
            if max_weight_child is None or max_weight_child.max_weight < cur.max_weight:
                break
            else:
                cur = max_weight_child
        if cur.is_root:
            return None
        selected_node: MCTSTreeNode = cur
        selected_node.pick()
        selected_result: ResultPath = selected_node.to_result()
        return selected_result

