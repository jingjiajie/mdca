import random
from typing import Iterable

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from analyzer.Index import Index, IndexLocations
from analyzer.commons import calc_weight, Value

if TYPE_CHECKING:
    from MCTSTree import MCTSTree, ColumnValueWeights


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
        self.children: list[MCTSTreeNode] | None = None
        self.column: str | None = column
        self.value: str | None = value
        self.q_value: int = 0
        self.locations: IndexLocations
        self.depth: int
        self.full_visited_flag: bool = False
        self._error_count: int = -1
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        self._init_location()

    @property
    def count(self) -> int:
        return self.locations.count

    @property
    def error_count(self) -> int:
        if self._error_count != -1:
            return self._error_count
        else:
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
        return calc_weight(self.depth, self.error_coverage, self.error_rate, self.tree.data_index.total_error_rate)

    @property
    def is_root(self) -> bool:
        return self.column is None and self.value is None

    def _init_location(self):
        if self.is_root:
            index: Index = self.tree.data_index
            self.locations = IndexLocations(index, np.ones(index.total_count, dtype=bool))
        else:
            self_loc: IndexLocations = self.tree.data_index.get_locations(self.column, self.value)
            self.locations = self.parent.locations & self_loc

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
        children: list[MCTSTreeNode] = []
        columns_after: list[str] = self.tree.data_index.get_columns_after(self.column)
        index: Index = self.tree.data_index
        for col in columns_after:
            values: Iterable = index.get_values_by_column(col)
            for val in values:
                loc: IndexLocations = index.get_locations(col, val)
                if loc.count < self.tree.min_error_count:
                    continue
                child = MCTSTreeNode(self.tree, self, col, val)
                if child.count < self.tree.min_error_count:
                    continue
                if child.error_count < self.tree.min_error_count:
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

    def simulate(self, simulate_times: int, max_simulate_depth: int = 10):
        """
        Update self.q_value as result
        :param max_simulate_depth: max depth of one simulation
        :param simulate_times: times of simulation
        """
        index: Index = self.tree.data_index
        total_error_loc: IndexLocations = index.get_locations(index.target_column, index.target_value)
        columns_after = self.tree.data_index.get_columns_after(self.column)
        max_weight: float = calc_weight(self.depth, self.error_coverage, self.error_rate, index.total_error_rate)
        for epoch in range(0, simulate_times):
            cur_locations: IndexLocations = self.locations
            all_selected_col_idx: list[int] = []
            all_selected_items = []  # todo temp
            while len(all_selected_col_idx) < min(len(columns_after), max_simulate_depth):
                next_col: str = self._select_next_column(columns_after, all_selected_col_idx) # todo 不要单独选列
                column_value_weights: ColumnValueWeights = self.tree._get_value_weights_by_column(next_col)
                if column_value_weights.max_weight == 0:
                    break
                values: list[Value | pd.Interval] = column_value_weights.values
                weights_normalized: np.ndarray = column_value_weights.weights_normalized
                selected_val_idx: int = np.random.choice(len(values), size=1, p=weights_normalized)[0]
                selected_val: Value | pd.Interval = values[selected_val_idx]
                all_selected_items.append(next_col + '=' + str(selected_val))
                selected_val_loc: IndexLocations = index.get_locations(next_col, selected_val)
                cur_locations = cur_locations & selected_val_loc
                if cur_locations.count < self.tree.min_error_count:
                    break
                cur_error_locations = cur_locations & total_error_loc
                if cur_error_locations.count < self.tree.min_error_count:
                    break
                error_coverage: float = cur_error_locations.count / total_error_loc.count
                error_rate: float = cur_error_locations.count / cur_locations.count
                cur_weight: float = calc_weight(1, error_coverage, error_rate, index.total_error_rate)
                if cur_weight > max_weight:
                    max_weight = cur_weight
            print(" --- epoch ", epoch, ':\t', ",".join(all_selected_items))
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
