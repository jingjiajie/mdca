import random

from Index import Index, IndexLocationList
from utils import bin_search_nearest_lower

# TODO 动态自适应
SIMULATE_TIMES: int = 500


class MCTSTreeNode:

    def __init__(self, parent: 'MCTSTreeNode', index: Index, column: str, value: str):
        self._index: Index = index
        self.parent: MCTSTreeNode = parent
        self.children: list[MCTSTreeNode] = []
        self.visit_count = 0
        self.column: str = column
        self.value: str = value
        self.influence_count: int

    def _get_depth(self):
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
            next_col_idx_pos: int = bin_search_nearest_lower(already_selected_col_idx, next_col_idx, last_col_idx_pos)
            if next_col_idx_pos == last_col_idx_pos:
                already_selected_col_idx.insert(next_col_idx_pos + 1, next_col_idx)
                return columns[next_col_idx]
            else:
                skip_count: int = next_col_idx_pos - last_col_idx_pos
                next_col_idx += skip_count

    def simulate(self, threshold: int) -> int:
        self_depth: int = self._get_depth()
        max_layer: int = self_depth
        columns_after = self._index.get_columns_after(self.column)
        for epoch in range(0, SIMULATE_TIMES):
            cur_locations: IndexLocationList = self._index.get_locations(self.column, self.value)
            all_selected_col_idx: list[int] = []
            while len(all_selected_col_idx) < len(columns_after):
                next_col: str = self._select_next_column(columns_after, all_selected_col_idx)
                selected_val = self._index.random_select_value_by_freq(next_col)
                locations: IndexLocationList = self._index.get_locations(next_col, selected_val)
                cur_locations = cur_locations.intersect(locations)
                if cur_locations.count < threshold:
                    depth: int = self_depth + len(all_selected_col_idx) - 1
                    if depth > max_layer:
                        max_layer = depth
                    break
            if len(all_selected_col_idx) == len(columns_after):
                return len(columns_after)
        return max_layer
