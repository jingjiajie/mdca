import random
from typing import Iterable

from analyzer.Index import IndexLocationList, IndexLocation

from typing import TYPE_CHECKING
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
        self.visit_count = 0
        self.q_value: int = 0
        self.locations: IndexLocationList
        self._init_location()

    @property
    def influence_count(self) -> int:
        return self.locations.count

    @property
    def is_root(self) -> bool:
        return self.column is None and self.value is None

    def _init_location(self):
        parent_loc: IndexLocationList
        if self.parent is None:
            parent_loc = IndexLocationList(IndexLocation(0, self.tree.data_index.total_count - 1))
        else:
            parent_loc = self.parent.locations

        if self.column is not None and self.value is not None:
            self_loc = self.tree.data_index.get_locations(self.column, self.value)
            self.locations = parent_loc.intersect(self_loc)
        else:
            self.locations = parent_loc

    @property
    def depth(self):
        node: MCTSTreeNode = self
        depth: int = 0
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth

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
        self.visit_count += 1
        if self.children is None or len(self.children) == 0:
            return self
        # TODO 快速查找UCB最大的child
        search_list: list[int] = []
        total: int = 0
        for c in self.children:
            total += c.q_value
            search_list.append(c.q_value)
        search_list.sort()
        rand: int = random.randint(0, total)
        idx: int = _bin_search_nearest_lower_int(search_list, rand)
        selected_child = self.children[idx]
        return selected_child.select()

    def expand(self):
        children: list[MCTSTreeNode] = []
        columns_after: list[str] = self.tree.data_index.get_columns_after(self.column)
        for col in columns_after:
            values: Iterable = self.tree.data_index.get_values_by_column(col)
            for val in values:
                child = MCTSTreeNode(self.tree, self, col, val)  # TODO 不满足influence_count不要创建node
                if child.influence_count >= self.tree.threshold:
                    # bin_search_nearest_lower(children, lambda c: c.)
                    children.append(child)  # TODO 按UCB大小排序
        self.children = children

    def simulate(self, simulate_times: int):
        """
        Update self.q_value as result
        :param simulate_times: times of simulation
        :param threshold: influenced lines
        """
        self_depth: int = self.depth
        max_layer: int = self_depth
        columns_after = self.tree.data_index.get_columns_after(self.column)
        for epoch in range(0, simulate_times):
            cur_locations: IndexLocationList = self.locations
            all_selected_col_idx: list[int] = []
            while len(all_selected_col_idx) < len(columns_after):
                next_col: str = self._select_next_column(columns_after, all_selected_col_idx)
                selected_val = self.tree.data_index.random_select_value_by_freq(next_col)
                locations: IndexLocationList = self.tree.data_index.get_locations(next_col, selected_val)
                cur_locations = cur_locations.intersect(locations)
                if cur_locations.count < self.tree.threshold:
                    depth: int = self_depth + len(all_selected_col_idx) - 1
                    if depth > max_layer:
                        max_layer = depth
                    break
            if len(all_selected_col_idx) == len(columns_after):
                return len(columns_after)
        self.q_value = max_layer

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
        while cur.parent is not None:
            if len(cur.parent.children) > 1:
                cur.parent.children.remove(cur)  # TODO 性能
                return
            else:
                cur = cur.parent
        # Reached root node, almost impossible
        cur.children = []

    def check(self) -> bool:
        included_cols: dict[str, bool] = {}
        cur: MCTSTreeNode = self
        while cur.parent is not None:
            included_cols[cur.column] = True
            cur = cur.parent
        columns_before: list[str] = self.tree.data_index.get_columns_before(self.column)
        for col in columns_before:
            if col in included_cols:
                continue
            # TODO 频率从高到低
            values: Iterable = self.tree.data_index.get_values_by_column(col)
            for val in values:
                val_loc: IndexLocationList = self.tree.data_index.get_locations(col, val)
                intersect_loc: IndexLocationList = self.locations.intersect(val_loc)
                if intersect_loc.count >= self.tree.threshold:
                    return False
        return True
