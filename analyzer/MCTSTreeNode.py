import weakref
from enum import Enum

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from analyzer.Index import Index, IndexLocations
from analyzer.ResultPath import ResultPath, ResultItem
from analyzer.commons import calc_weight_fairness, Value, calc_weight_distribution

if TYPE_CHECKING:
    from MCTSTree import MCTSTree


class TreeNodeState(Enum):
    DEFAULT_STATE: int = 0
    FULL_VISITED_STATE: int = 1
    PICKED_STATE: int = 2
    FULL_PICKED_STATE: int = 3


class MCTSTreeNode:

    def __init__(self, tree: 'MCTSTree', parent: 'MCTSTreeNode | None', column: str | None, value: str | None,
                 locations: IndexLocations):
        self._tree_ref = weakref.ref(tree)
        self.parent: MCTSTreeNode = parent
        self.children: dict[str, MCTSTreeNode] | None = None
        self.column: str | None = column
        self.value: str | None = value
        self.max_weight: float = 0
        self._self_weight: float = -1
        self.locations: IndexLocations = locations
        self.state: TreeNodeState = TreeNodeState.DEFAULT_STATE
        self._target_count: int = -1
        self.depth: int
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    @property
    def tree(self) -> 'MCTSTree':
        return self._tree_ref()

    @property
    def count(self) -> int:
        return self.locations.count

    @property
    def target_count(self) -> int:
        if self._target_count == -1:
            index: Index = self.tree.data_index
            self._target_count = (self.locations & index.total_target_locations).count
        return self._target_count

    @property
    def target_coverage(self) -> float:
        index: Index = self.tree.data_index
        return self.target_count / index.total_target_count

    @property
    def total_coverage(self) -> float:
        index: Index = self.tree.data_index
        return self.count / index.total_count

    @property
    def target_rate(self) -> float:
        return self.target_count / self.count

    @property
    def weight(self) -> float:
        if self._self_weight == -1:
            if self.tree.search_mode == 'fairness':
                self._self_weight = calc_weight_fairness(
                    self.depth, self.target_coverage, self.target_rate, self.tree.data_index.total_target_rate)
            elif self.tree.search_mode == 'distribution':
                col_values: dict[str, Value | pd.Interval] = {}
                node: MCTSTreeNode = self
                while node is not None:
                    if node.column is not None:
                        col_values[node.column] = node.value
                    node = node.parent
                baseline_coverage: float = self.tree.data_index.get_column_combination_coverage_baseline(col_values)
                self._self_weight = calc_weight_distribution(self.depth, self.total_coverage, baseline_coverage)
        return self._self_weight

    @property
    def is_root(self) -> bool:
        return self.column is None and self.value is None

    def select(self) -> 'MCTSTreeNode | None':
        if self.children is None:
            return self
        # TODO 性能优化
        non_full_visited_children: list[MCTSTreeNode] = (
            list(filter(lambda child: child.state == TreeNodeState.DEFAULT_STATE, self.children.values())))
        if len(non_full_visited_children) == 0:
            return self
        weights: np.ndarray[np.float64] = np.ndarray(len(non_full_visited_children), dtype=np.float64)
        for i in range(len(non_full_visited_children)):
            child: MCTSTreeNode = non_full_visited_children[i]
            weights[i] = child.max_weight
        weights_normalized: np.ndarray[np.float64] = weights/weights.sum()
        selected_child: MCTSTreeNode = np.random.choice(non_full_visited_children, size=1, p=weights_normalized)[0]
        return selected_child.select()

    def expand(self):
        index: Index = self.tree.data_index
        children: dict[str, MCTSTreeNode] = {}
        columns_after: list[str] = self.tree.data_index.get_columns_after(self.column)
        is_fairness: bool = self.tree.search_mode == 'fairness'
        for col in columns_after:
            value_dict: dict[Value | pd.Interval, IndexLocations] =\
                self.tree._get_candidate_values_by_column(col)
            for val, val_loc in value_dict.items():

                fast_predict_intersect_count: bool | None
                if is_fairness:
                    fast_predict_intersect_count = Index.fast_predict_bool_intersect_count(
                        [self.locations, val_loc, index.total_target_locations])
                else:
                    fast_predict_intersect_count = Index.fast_predict_bool_intersect_count(
                        [self.locations, val_loc])
                if (fast_predict_intersect_count is not None and
                        fast_predict_intersect_count < self.tree.min_count * 0.5):
                    continue
                child_loc: IndexLocations = self.locations & val_loc
                child = MCTSTreeNode(self.tree, self, col, val, child_loc)
                if child.count < self.tree.min_count:
                    continue
                if is_fairness and child.target_count < self.tree.min_count:
                    continue
                if child.weight == 0:
                    continue
                children[str(child)] = child
        self.children = children
        if self.children is not None and len(self.children) == 0:
            cur = self
            while cur is not None:
                if all(map(lambda c: c.state == TreeNodeState.FULL_VISITED_STATE, cur.children.values())):
                    cur.state = TreeNodeState.FULL_VISITED_STATE
                    cur = cur.parent
                else:
                    break

    def simulate(self):
        self.max_weight = self.weight

    def back_propagate(self):
        cur: MCTSTreeNode = self.parent
        while cur is not None:
            if cur.max_weight < self.max_weight:
                cur.max_weight = self.max_weight
            cur = cur.parent

    def __str__(self):
        if self.is_root:
            return "[MCTS Root]"
        else:
            return f"{self.column}={self.value}"

    def path(self):
        path = []
        node: MCTSTreeNode = self
        while node is not None:
            if node.column is not None:
                path.append(str(node))
            node = node.parent
        path.reverse()
        return "[" + ", ".join(path) + "]"

    def pick(self):
        if (len(self.children) == 0 or
                all(map(lambda c: c.state == TreeNodeState.FULL_PICKED_STATE, self.children.values()))):
            self.state = TreeNodeState.FULL_PICKED_STATE
        else:
            self.state = TreeNodeState.PICKED_STATE

        cur: MCTSTreeNode = self
        while cur is not None:
            child_max_weight: float = 0
            for child in cur.children.values():
                if child.max_weight > child_max_weight:
                    child_max_weight = child.max_weight
            cur.max_weight = child_max_weight
            cur = cur.parent

    def to_result(self) -> ResultPath:
        result_items: list[ResultItem] = []
        cur = self
        while cur.parent is not None:
            result_items.append(
                ResultItem(cur.column, self.tree.column_info[cur.column].column_type, cur.value,
                           self.tree.data_index.get_locations(cur.column, cur.value)))
            cur = cur.parent
        result_items.reverse()
        result_path: ResultPath = ResultPath(result_items, self.locations, self.tree.search_mode)
        return result_path
