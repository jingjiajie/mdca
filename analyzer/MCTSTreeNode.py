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
        self._tree_weak_ref = weakref.ref(tree)
        self.parent: MCTSTreeNode = parent
        self.children: list[MCTSTreeNode] | None = None
        self.column: str | None = column
        self.value: str | None = value
        self.q_value: int = 0
        self.locations: IndexLocations = locations
        self.full_visited_flag: bool = False
        self._error_count: int = -1

    @property
    def tree(self) -> 'MCTSTree':
        return self._tree_weak_ref()

    @property
    def count(self) -> int:
        return self.locations.count

    @property
    def error_count(self) -> int:
        if self._error_count == -1:
            index: Index = self.tree.data_index
            self._error_count = (self.locations & index.total_error_locations).count
        return self._error_count

    @property
    def error_coverage(self) -> float:
        index: Index = self.tree.data_index
        return self.error_count / index.total_error_locations.count

    @property
    def error_rate(self) -> float:
        return self.error_count / self.count

    @property
    def weight(self) -> float:
        return calc_weight(1, self.error_coverage, self.error_rate, self.tree.data_index.total_error_rate)

    @property
    def is_root(self) -> bool:
        return self.column is None and self.value is None

    def select(self) -> 'MCTSTreeNode | None':
        if self.children is None:
            return self
        # TODO 性能优化
        non_full_visited_children = list(filter(lambda child: not child.full_visited_flag, self.children))
        if len(non_full_visited_children) == 0:
            return self  # Should be root
        weights: np.ndarray[np.float64] = np.ndarray(len(non_full_visited_children), dtype=np.float64)
        for i in range(len(non_full_visited_children)):
            child: MCTSTreeNode = non_full_visited_children[i]
            weights[i] = child.q_value
        weights_normalized: np.ndarray[np.float64] = weights/weights.sum()
        selected_child: MCTSTreeNode = np.random.choice(non_full_visited_children, size=1, p=weights_normalized)[0]
        return selected_child.select()

    def expand(self):
        index: Index = self.tree.data_index
        children: list[MCTSTreeNode] = []
        columns_after: list[str] = self.tree.data_index.get_columns_after(self.column)
        # try_count = 0
        # try_error_count = 0
        # fast_predict_fail_count = 0
        for col in columns_after:
            value_dict: dict[Value | pd.Interval, IndexLocations] =\
                self.tree._get_values_satisfy_min_error_coverage_by_column(col)
            for val, val_loc in value_dict.items():
                # try_count += 1
                fast_predict_intersect_count: bool | None = Index.fast_predict_bool_intersect_count(
                    [self.locations, val_loc, index.total_error_locations])
                if (fast_predict_intersect_count is not None and
                        fast_predict_intersect_count < self.tree.min_error_count * 0.8):
                    # fast_predict_fail_count += 1
                    continue
                child_loc: IndexLocations = self.locations & val_loc
                child = MCTSTreeNode(self.tree, self, col, val, child_loc)
                if child.count < self.tree.min_error_count:
                    continue
                # try_error_count += 1
                if child.error_count < self.tree.min_error_count:
                    continue
                children.append(child)
        # print("Tried count: %d, pred_fail: %d, try_error_count: %d, final_pass: %d" %
        #       (try_count, fast_predict_fail_count, try_error_count, len(children)))
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
        self.q_value = calc_weight(1, self.error_coverage, self.error_rate,
                                   self.tree.data_index.total_error_rate)

    def back_propagate(self):
        cur: MCTSTreeNode = self.parent
        while cur is not None:
            if cur.q_value < self.q_value:
                cur.q_value = self.q_value
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
                cur.q_value = cur.weight
            else:
                max_q_child: MCTSTreeNode = cur.children[0]
                for child in cur.children:
                    if child.q_value > max_q_child.q_value:
                        max_q_child = child
                cur.q_value = max_q_child.q_value
            cur = cur.parent
