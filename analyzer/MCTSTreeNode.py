import weakref

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from analyzer.Index import Index, IndexLocations, IndexLocationType
from analyzer.commons import calc_weight, Value

if TYPE_CHECKING:
    from MCTSTree import MCTSTree


class MCTSTreeNode:

    def __init__(self, tree: 'MCTSTree', parent: 'MCTSTreeNode | None', column: str | None, value: str | None,
                 locations: IndexLocations):
        self._tree_ref = weakref.ref(tree)
        self.parent: MCTSTreeNode = parent
        self.children: list[MCTSTreeNode] | None = None
        self.column: str | None = column
        self.value: str | None = value
        self.max_weight: float = 0
        self._self_weight: float = -1
        self.locations: IndexLocations = locations
        self.full_visited_flag: bool = False
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
        return self.target_count / index.total_target_locations.count

    @property
    def target_rate(self) -> float:
        return self.target_count / self.count

    @property
    def weight(self) -> float:
        if self._self_weight == -1:
            self._self_weight = calc_weight(
                self.depth, self.target_coverage, self.target_rate, self.tree.data_index.total_target_rate)
        return self._self_weight

    @property
    def is_root(self) -> bool:
        return self.column is None and self.value is None

    def select(self) -> 'MCTSTreeNode | None':
        if self.children is None:
            return self
        # TODO 性能优化
        non_full_visited_children = list(filter(lambda child: not child.full_visited_flag, self.children))
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
        children: list[MCTSTreeNode] = []
        columns_after: list[str] = self.tree.data_index.get_columns_after(self.column)
        for col in columns_after:
            value_dict: dict[Value | pd.Interval, IndexLocations] =\
                self.tree._get_values_satisfy_min_target_coverage_by_column(col)
            for val, val_loc in value_dict.items():
                fast_predict_intersect_count: bool | None = Index.fast_predict_bool_intersect_count(
                    [self.locations, val_loc, index.total_target_locations])
                if (fast_predict_intersect_count is not None and
                        fast_predict_intersect_count < self.tree.min_target_count * 0.5):
                    continue
                child_loc: IndexLocations = self.locations & val_loc
                child = MCTSTreeNode(self.tree, self, col, val, child_loc)
                if child.count < self.tree.min_target_count:
                    continue
                if child.target_count < self.tree.min_target_count:
                    continue
                if child.weight == 0:
                    continue
                children.append(child)
        self.children = children
        if self.children is not None and len(self.children) == 0:
            cur = self
            while cur is not None:
                if all(map(lambda c: c.full_visited_flag, cur.children)):
                    cur.full_visited_flag = True
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
        cur: MCTSTreeNode = self
        while True:
            if len(cur.parent.children) > 1 or cur.parent.is_root:
                cur.parent.children.remove(cur)
                cur = cur.parent
                break
            else:
                cur = cur.parent

        while cur is not None:
            if len(cur.children) == 0:
                cur.max_weight = cur.weight
            else:
                max_weight_child: MCTSTreeNode = cur.children[0]
                for child in cur.children:
                    if child.max_weight > max_weight_child.max_weight:
                        max_weight_child = child
                cur.max_weight = max(max_weight_child.max_weight, cur.weight)
            cur = cur.parent
