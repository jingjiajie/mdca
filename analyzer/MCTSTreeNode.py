import random
from typing import Iterable

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from analyzer.Index import Index
from analyzer.commons import calc_weight

if TYPE_CHECKING:
    from MCTSTree import MCTSTree


def _bin_search_nearest_lower_int(search_list_asc: list[int], value: int, low: int | None = None, high: int | None = None) -> int:
    """
    :param high: search end index
    :param low: search start index
    :param search_list_asc: list to search
    :param value: value to search
    :return: -1 if value is the lowest, otherwise the index of nearest lower element
    """
    if len(search_list_asc) == 0:
        return -1
    if low is None or low < 0:
        low = 0
    if high is None or high >= len(search_list_asc):
        high = len(search_list_asc) - 1
    while low < high:
        mid: int = int((low + high) / 2)
        cur: int = search_list_asc[mid]
        if cur < value:
            if mid == low:
                low += 1
            else:
                low = mid
        elif cur > value:
            if mid == high:
                high -= 1
            else:
                high = mid
        else:  # cur == rand
            return mid
    if search_list_asc[low] > value:
        low -= 1
    return low


class MCTSTreeNode:

    def __init__(self, tree: 'MCTSTree', parent: 'MCTSTreeNode | None', column: str | None, value: str | None):
        self.tree = tree
        self.parent: MCTSTreeNode = parent
        self.children: list[MCTSTreeNode] | None = None  # TODO 改成跳表
        self.column: str | None = column
        self.value: str | None = value
        self.q_value: int = 0
        self.locations: np.ndarray | None
        self.depth: int
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        self._init_location()

    @property
    def count(self) -> int:
        if self.is_root:
            return self.tree.data_index.total_count
        else:
            return self.locations.sum()

    @property
    def error_count(self) -> int:
        index: Index = self.tree.data_index
        err_loc: pd.Series = index.get_locations(index.target_column, index.target_value)
        if self.is_root:
            return err_loc.sum()
        else:
            return (self.locations & err_loc).sum()

    @property
    def error_coverage(self) -> float:
        index: Index = self.tree.data_index
        err_loc: pd.Series = index.get_locations(index.target_column, index.target_value)
        return self.error_count / err_loc.sum()

    @property
    def error_rate(self) -> float:
        return self.error_count / self.count

    @property
    def weight(self) -> float:
        return calc_weight(self.depth, self.error_coverage, self.error_rate, self.tree.data_index.total_error_rate)

    @property
    def is_root(self) -> bool:
        return self.column is None and self.value is None

    def _init_location(self):
        if self.is_root:
            self.locations = None
        else:
            self_loc: np.ndarray = self.tree.data_index.get_locations(self.column, self.value)
            parent_loc: np.ndarray | None = self.parent.locations
            if parent_loc is None:
                self.locations = self_loc
            else:
                self.locations = parent_loc & self_loc

    def _select_next_column(self, columns: list[str], already_selected_col_idx: list[int]) -> str:
        rand = random.randint(0, len(columns) - 1 - len(already_selected_col_idx))
        next_col_idx: int = rand
        next_col_idx_pos: int = -1
        last_col_idx_pos: int
        while True:
            last_col_idx_pos = next_col_idx_pos
            next_col_idx_pos: int = _bin_search_nearest_lower_int(already_selected_col_idx, next_col_idx, last_col_idx_pos)
            if next_col_idx_pos == last_col_idx_pos:
                already_selected_col_idx.insert(next_col_idx_pos + 1, next_col_idx)
                return columns[next_col_idx]
            else:
                skip_count: int = next_col_idx_pos - last_col_idx_pos
                next_col_idx += skip_count

    def select(self) -> 'MCTSTreeNode | None':
        if self.children is None or len(self.children) == 0:
            return self
        # TODO 性能优化
        children_weights: np.ndarray[np.float64] = np.ndarray(len(self.children), dtype=np.float64)
        for i in range(len(self.children)):
            child: MCTSTreeNode = self.children[i]
            children_weights[i] = child.q_value
        weights_normalized: np.ndarray[np.float64] = children_weights/children_weights.sum()
        selected_child: MCTSTreeNode = np.random.choice(self.children, size=1, p=weights_normalized)[0]
        return selected_child.select()

    def expand(self):
        children: list[MCTSTreeNode] = []
        columns_after: list[str] = self.tree.data_index.get_columns_after(self.column)
        for col in columns_after:
            values: Iterable = self.tree.data_index.get_values_by_column(col)
            for val in values:
                child = MCTSTreeNode(self.tree, self, col, val)  # TODO 不满足influence_count不要创建node
                if child.error_coverage >= self.tree.min_error_coverage:
                    children.append(child)
        self.children = children

    def simulate(self, simulate_times: int, max_simulate_depth: int = 10):
        """
        Update self.q_value as result
        :param max_simulate_depth: max depth of one simulation
        :param simulate_times: times of simulation
        """
        index: Index = self.tree.data_index
        columns_after = self.tree.data_index.get_columns_after(self.column)
        err_loc: pd.Series = index.get_locations(index.target_column, index.target_value)
        max_weight: float = calc_weight(self.depth, self.error_coverage, self.error_rate, index.total_error_rate)
        for epoch in range(0, simulate_times):
            cur_locations: pd.Series = self.locations
            all_selected_col_idx: list[int] = []
            while len(all_selected_col_idx) < min(len(columns_after), max_simulate_depth):
                next_col: str = self._select_next_column(columns_after, all_selected_col_idx)
                # todo 小于min_error_coverage的value不要返回
                values: list[str] = list(index.get_values_by_column(next_col))
                weights: np.ndarray[np.float64] = np.ndarray(shape=len(values), dtype=np.float64)
                i: int = 0
                for val in values:
                    val: str
                    new_loc: pd.Series = index.get_locations(next_col, val) & cur_locations
                    new_loc_count: int = new_loc.sum()
                    if new_loc_count == 0:
                        weight = 0
                    else:
                        new_error_loc: pd.Series = new_loc & err_loc
                        new_error_count: int = new_error_loc.sum()
                        err_coverage: float = new_error_count / err_loc.sum()
                        if err_coverage < self.tree.min_error_coverage:
                            weights[i] = 0
                            i += 1
                            continue
                        err_rate: float = new_error_count / new_loc_count
                        weight: float = calc_weight(self.depth + len(all_selected_col_idx),
                                                    err_coverage, err_rate, index.total_error_rate)
                        if np.isnan(weight) or np.isnan(np.float64(weight)):
                            pass
                    weights[i] = weight
                    if weight > max_weight:
                        max_weight = weight
                    i += 1
                if np.all(weights == 0):
                    break
                weights_normalized: np.ndarray[np.float64] = weights / weights.sum()
                selected_val: str = np.random.choice(values, size=1, p=weights_normalized)[0]
                selected_val_loc: pd.Series = index.get_locations(next_col, selected_val)
                cur_locations = cur_locations & selected_val_loc
        self.q_value = max_weight

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
                cur.parent.children.remove(cur)  # TODO 性能
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
