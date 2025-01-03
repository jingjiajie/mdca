import math
import random
from typing import Iterable

from Index import Index, IndexLocationList, IndexLocation
from utils import bin_search_nearest_lower_int

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from MCTSTree import MCTSTree

# TODO 动态自适应
SIMULATE_TIMES: int = 500

UCB_C = 2  # TODO 寻找合适的值
UCB_ALPHA = 0.001  # TODO 寻找合适的值


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
            next_col_idx_pos: int = bin_search_nearest_lower_int(already_selected_col_idx, next_col_idx, last_col_idx_pos)
            if next_col_idx_pos == last_col_idx_pos:
                already_selected_col_idx.insert(next_col_idx_pos + 1, next_col_idx)
                return columns[next_col_idx]
            else:
                skip_count: int = next_col_idx_pos - last_col_idx_pos
                next_col_idx += skip_count

    def _calculate_ucb(self) -> float:
        if self.parent is None:  # No UCB for root node
            return 0
        return self.q_value + UCB_C * math.sqrt(math.log(self.parent.visit_count, math.e) / (self.visit_count + UCB_ALPHA))

    def select(self) -> 'MCTSTreeNode | None':
        self.visit_count += 1
        if self.children is None or len(self.children) == 0:
            return self
        # TODO 快速查找UCB最大的child
        max_ucb_child: MCTSTreeNode = self.children[0]
        max_ucb: float = self.children[0]._calculate_ucb()
        for c in self.children:
            ucb: float = c._calculate_ucb()
            if ucb > max_ucb:
                max_ucb_child = c
                max_ucb = ucb
        return max_ucb_child.select()

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

    def simulate(self):
        """
        Update self.q_value as result
        :param threshold: influenced lines
        """
        self_depth: int = self.depth
        max_layer: int = self_depth
        columns_after = self.tree.data_index.get_columns_after(self.column)
        for epoch in range(0, SIMULATE_TIMES):
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
        return f"{self.column}={self.value}"

    def path(self):
        path = []
        node: MCTSTreeNode = self
        while node is not None:
            if node.column is not None:
                path.append(str(node))
            node = node.parent
        path.reverse()
        return "[" + ",".join(path) + "]"

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
