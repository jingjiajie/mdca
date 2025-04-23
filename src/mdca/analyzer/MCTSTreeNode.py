import weakref
from enum import Enum

from typing import TYPE_CHECKING

import numpy as np

from mdca.analyzer.Index import Index, IndexLocations
from mdca.analyzer.ResultPath import ResultPath, ResultItem, BinResultValue
from mdca.analyzer.commons import calc_weight_fairness, Value, calc_weight_error, ColumnInfo

if TYPE_CHECKING:
    from MCTSTree import MCTSTree


class TreeNodeVisitState(Enum):
    NOT_VISITED: int = 0
    VISITED: int = 1
    FULL_VISITED: int = 2


class TreeNodeCutRange:

    def __init__(self, lower_cut_point_id: int | None, upper_cut_point_id: int | None):
        self.lower_cut_point_id: int | None = lower_cut_point_id
        self.upper_cut_point_id: int | None = upper_cut_point_id

    def copy(self) -> 'TreeNodeCutRange':
        return TreeNodeCutRange(self.lower_cut_point_id, self.upper_cut_point_id)

    def __str__(self) -> str:
        s: str = f"({self.lower_cut_point_id}, {self.upper_cut_point_id})"
        return s


class MCTSTreeNode:

    def __init__(self, tree: 'MCTSTree', parent: 'MCTSTreeNode | None', column: str | None, value: Value | None,
                 is_cut_point: bool, cut_point_gte: bool | None, cut_point_id: int | None, locations: IndexLocations,
                 derived_from: 'MCTSTreeNode | None'):
        self._tree_ref = weakref.ref(tree)
        self.parent: MCTSTreeNode = parent
        self.children: dict[str, MCTSTreeNode] | None = None
        self.column: str | None = column
        self.value: str | None = value
        self.is_cut_point: bool = is_cut_point
        self.cut_point_gte: bool | None = cut_point_gte
        self.cut_point_id: int | None = cut_point_id
        self.max_weight: float = 0
        self._self_weight: float = -1
        self.locations: IndexLocations = locations
        self.derived_from: MCTSTreeNode | None = derived_from
        self.visit_state: TreeNodeVisitState = TreeNodeVisitState.NOT_VISITED
        self._target_count: int = -1
        self.depth: int
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

        self.existing_cut_ranges: dict[str, TreeNodeCutRange] = {}
        if parent is not None:
            for k, v in parent.existing_cut_ranges.items():
                self.existing_cut_ranges[k] = v.copy()
            if self.is_cut_point:
                if column not in self.existing_cut_ranges:
                    self.existing_cut_ranges[column] = TreeNodeCutRange(None, None)
                existing_cut_range: TreeNodeCutRange = self.existing_cut_ranges[column]
                if cut_point_gte and (existing_cut_range.lower_cut_point_id is None or
                                      cut_point_id > existing_cut_range.lower_cut_point_id):
                    self.existing_cut_ranges[column].lower_cut_point_id = cut_point_id
                elif not cut_point_gte and (existing_cut_range.upper_cut_point_id is None or
                                            cut_point_id < existing_cut_range.upper_cut_point_id):
                    self.existing_cut_ranges[column].upper_cut_point_id = cut_point_id

        # col -> lower_bin_cut_point
        self.empty_bins: dict[str, set[int]] = {}

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
    def coverage(self) -> float:
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
                    self.depth, self.coverage, self.target_rate, self.tree.data_index.total_target_rate)
            elif self.tree.search_mode == 'error':
                self._self_weight = calc_weight_error(
                    self.depth, self.target_coverage, self.target_rate, self.tree.data_index.total_target_rate)
        return self._self_weight

    @property
    def is_root(self) -> bool:
        return self.column is None and self.value is None

    def sync_empty_bins_from_parent(self):
        if self.parent is not None:
            for k, v in self.parent.empty_bins.items():
                self.empty_bins[k] = set(v)

    def select(self) -> 'MCTSTreeNode | None':
        if self.children is None:
            return self
        # TODO 性能优化
        non_full_visited_children: list[MCTSTreeNode] = (
            list(filter(lambda child: child.visit_state != TreeNodeVisitState.FULL_VISITED, self.children.values())))
        if len(non_full_visited_children) == 0:
            return self
        weights: np.ndarray[np.float64] = np.ndarray(len(non_full_visited_children), dtype=np.float64)
        for i in range(len(non_full_visited_children)):
            child: MCTSTreeNode = non_full_visited_children[i]
            if child.derived_from is not None and child.visit_state == TreeNodeVisitState.NOT_VISITED:
                base_node: MCTSTreeNode = child.derived_from
                weights[i] = base_node.max_weight
            else:
                weights[i] = child.max_weight
        weights_normalized: np.ndarray[np.float64] = weights / weights.sum()
        selected_child: MCTSTreeNode = np.random.choice(non_full_visited_children, size=1, p=weights_normalized)[0]
        selected_child.visit_state = TreeNodeVisitState.VISITED
        return selected_child.select()

    def expand(self):
        index: Index = self.tree.data_index
        children: dict[str, MCTSTreeNode] = {}

        columns_after: list[str] = self.tree.data_index.get_columns_after(self.column)
        candidate_columns: list[str] = columns_after
        if (not self.is_root and self.tree.column_info[self.column].binning and
                not (isinstance(self.value, float) and np.isnan(self.value))):
            candidate_columns = [self.column] + columns_after
        for col in candidate_columns:
            col_info: ColumnInfo = self.tree.column_info[col]
            if col_info.binning:
                range_lower_cut_point_id: int = -1
                range_upper_cut_point_id: int = len(col_info.cut_points)
                if col in self.existing_cut_ranges:
                    if self.existing_cut_ranges[col].lower_cut_point_id is not None:
                        range_lower_cut_point_id = self.existing_cut_ranges[col].lower_cut_point_id
                    if self.existing_cut_ranges[col].upper_cut_point_id is not None:
                        range_upper_cut_point_id = self.existing_cut_ranges[col].upper_cut_point_id

                # Try nan value for the binning columns only before cutting
                if self.column != col:
                    na_loc: IndexLocations = index.get_cut_point_locations(col, True, None, None)
                    add_na_child: bool = True
                    if self.tree.min_count > 0:
                        if na_loc.count < self.tree.min_count:
                            add_na_child = False
                        else:
                            fast_predict_count: bool | None = Index.fast_predict_bool_intersect_count(
                                [self.locations, na_loc])
                            if (fast_predict_count is not None and
                                    fast_predict_count < self.tree.min_count * 0.5):
                                add_na_child = False
                    if self.tree.min_target_count > 0:
                        fast_predict_count: bool | None = Index.fast_predict_bool_intersect_count(
                            [self.locations, na_loc, index.total_target_locations])
                        if (fast_predict_count is not None and
                                fast_predict_count < self.tree.min_target_count * 0.5):
                            add_na_child = False
                    if add_na_child:
                        child_loc: IndexLocations = self.locations & na_loc
                        child = MCTSTreeNode(self.tree, self, col, np.nan, False, None, None, child_loc, None)
                        self._try_add_child(children, child)

                best_cut_point_child_gte: MCTSTreeNode | None = None
                best_cut_point_child_lt: MCTSTreeNode | None = None

                if range_upper_cut_point_id > range_lower_cut_point_id:
                    best_cut_point_child_gte: MCTSTreeNode = (
                        self._get_best_cut_point_child(col, True,
                                                       range_lower_cut_point_id, range_upper_cut_point_id))
                    best_cut_point_child_lt: MCTSTreeNode = \
                        self._get_best_cut_point_child(col, False,
                                                       range_lower_cut_point_id, range_upper_cut_point_id)

                best_cut_point_child: MCTSTreeNode | None = best_cut_point_child_gte
                if best_cut_point_child is None or (
                        best_cut_point_child_lt is not None and
                        best_cut_point_child_lt.weight > best_cut_point_child.weight):
                    best_cut_point_child = best_cut_point_child_lt

                if best_cut_point_child is not None and best_cut_point_child.weight > self.weight:
                    self._try_add_child(children, best_cut_point_child)
                    derived_child_loc: IndexLocations = \
                        (self.locations &
                         index.get_cut_point_locations(col, False,
                                                       best_cut_point_child.cut_point_id,
                                                       not best_cut_point_child.cut_point_gte))
                    derived_child: MCTSTreeNode = MCTSTreeNode(self.tree, self, col,
                                                                col_info.cut_points[best_cut_point_child.cut_point_id],
                                                                True,
                                                                not best_cut_point_child.cut_point_gte,
                                                                best_cut_point_child.cut_point_id,
                                                                derived_child_loc,
                                                                best_cut_point_child)
                    self._try_add_child(children, derived_child)
            else:
                value_dict: dict[Value, IndexLocations] = self.tree._get_categorical_candidate_values_by_column(col)
                for val, val_loc in value_dict.items():
                    if self.tree.min_count > 0:
                        fast_predict_count: bool | None = Index.fast_predict_bool_intersect_count(
                            [self.locations, val_loc])
                        if (fast_predict_count is not None and
                                fast_predict_count < self.tree.min_count * 0.5):
                            continue
                    if self.tree.min_target_count > 0:
                        fast_predict_count: bool | None = Index.fast_predict_bool_intersect_count(
                            [self.locations, val_loc, index.total_target_locations])
                        if (fast_predict_count is not None and
                                fast_predict_count < self.tree.min_target_count * 0.5):
                            continue
                    child_loc: IndexLocations = self.locations & val_loc
                    child = MCTSTreeNode(self.tree, self, col, val, False, None, None, child_loc, None)
                    self._try_add_child(children, child)
        for child in children.values():
            child.sync_empty_bins_from_parent()
        self.children = children
        if self.children is not None and len(self.children) == 0:
            cur = self
            while cur is not None:
                if all(map(lambda c: c.visit_state == TreeNodeVisitState.FULL_VISITED, cur.children.values())):
                    cur.visit_state = TreeNodeVisitState.FULL_VISITED
                    cur = cur.parent
                else:
                    break
            is_derived: bool = self.derived_from is not None
            if is_derived:
                del self.parent.children[str(self)]

    def _get_best_cut_point_child(self, col: str, gte: bool, lower_exclusive: int, upper_exclusive: int) \
            -> 'MCTSTreeNode':
        if upper_exclusive <= lower_exclusive:
            raise Exception('upper_exclusive must be greater than lower_exclusive')
        index: Index = self.tree.data_index
        col_info: ColumnInfo = self.tree.column_info[col]

        best_cut_point_child: MCTSTreeNode | None = None
        if gte:
            loop_range = range(lower_exclusive + 1, upper_exclusive)
        else:
            loop_range = range(upper_exclusive - 1, lower_exclusive, -1)
        last_count: int = -1

        for cut_point_id in loop_range:
            if col in self.empty_bins:
                if gte and (cut_point_id - 1) in self.empty_bins[col]:
                    continue
                elif not gte and cut_point_id in self.empty_bins[col]:
                    continue
            cut_loc: IndexLocations = index.get_cut_point_locations(col, False, cut_point_id, gte)
            child_loc: IndexLocations = self.locations & cut_loc
            child: MCTSTreeNode = MCTSTreeNode(self.tree, self, col, col_info.cut_points[cut_point_id],
                                               True, gte, cut_point_id, child_loc, None)
            if child.count == last_count:
                if gte:
                    self._add_empty_bin(col, cut_point_id - 1)
                else:
                    self._add_empty_bin(col, cut_point_id)
            last_count = child.count
            if self.tree.min_count > 0 and child.count < self.tree.min_count:
                break
            elif self.tree.min_target_count > 0 and child.target_count < self.tree.min_target_count:
                break
            if best_cut_point_child is None or child.weight > best_cut_point_child.weight:
                best_cut_point_child = child

        return best_cut_point_child

    def _add_empty_bin(self, col: str, bin_lower_cut_point_id: int):
        empty_bin_set: set[int]
        if col not in self.empty_bins:
            empty_bin_set = set()
            self.empty_bins[col] = empty_bin_set
        else:
            empty_bin_set = self.empty_bins[col]

        empty_bin_set.add(bin_lower_cut_point_id)

    def _try_add_child(self, children_map: dict[str, 'MCTSTreeNode'], child: 'MCTSTreeNode') -> bool:
        if child.count == 0 or child.count < self.tree.min_count:
            return False
        elif self.tree.target_column is not None and child.target_count < self.tree.min_target_count:
            return False

        is_derived_child: bool = child.derived_from is not None
        if is_derived_child and child.visit_state == TreeNodeVisitState.NOT_VISITED:
            children_map[str(child)] = child
            return True

        if child.weight == 0:
            return False
        children_map[str(child)] = child
        return True

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
        elif self.is_cut_point:
            if self.cut_point_gte:
                return f"{self.column}>={self.value}"
            else:
                return f"{self.column}<{self.value}"
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
        while not cur.parent.is_root:
            if len(cur.parent.children) == 1:
                if str(cur) not in cur.parent.children:
                    raise Exception(f'unexpected error, {cur.column} not in parent.children')
                cur = cur.parent
            else:
                break

        del cur.parent.children[str(cur)]

        while cur.parent is not None:
            child_max_weight: float = 0
            if cur.parent.children is not None:
                for child in cur.parent.children.values():
                    if child.max_weight > child_max_weight:
                        child_max_weight = child.max_weight
            cur.parent.max_weight = child_max_weight
            cur = cur.parent

    def to_result(self) -> ResultPath:
        index: Index = self.tree.data_index
        result_item_map: dict[str, ResultItem] = {}
        cur = self
        while cur.parent is not None:
            col_info: ColumnInfo = self.tree.column_info[cur.column]
            if col_info.binning:
                if isinstance(cur.value, float) and np.isnan(cur.value):
                    loc: IndexLocations = index.get_cut_point_locations(cur.column, True, None, None)
                    result_item_map[cur.column] = ResultItem(cur.column, col_info.column_type, cur.value, loc)
            else:
                loc: IndexLocations = index.get_categorical_locations(cur.column, cur.value)
                result_item_map[cur.column] = ResultItem(cur.column, col_info.column_type, cur.value, loc)
            cur = cur.parent

        for col, cut_range in self.existing_cut_ranges.items():
            col_info: ColumnInfo = self.tree.column_info[col]

            if cut_range.lower_cut_point_id is None and cut_range.upper_cut_point_id is None:
                raise Exception('Unexpected error')

            loc_gte_lower: IndexLocations | None = None
            lower_cut_point_value: Value | None = None
            if cut_range.lower_cut_point_id is not None:
                lower_cut_point_value = col_info.cut_points[cut_range.lower_cut_point_id]
                loc_gte_lower = index.get_cut_point_locations(col, False, cut_range.lower_cut_point_id, True)

            loc_lt_upper: IndexLocations | None = None
            upper_cut_point_value: Value | None = None
            if cut_range.upper_cut_point_id is not None:
                upper_cut_point_value = col_info.cut_points[cut_range.upper_cut_point_id]
                loc_lt_upper = index.get_cut_point_locations(col, False, cut_range.upper_cut_point_id, False)

            loc: IndexLocations
            if loc_gte_lower is None:
                loc = loc_lt_upper
            elif loc_lt_upper is None:
                loc = loc_gte_lower
            else:
                loc = loc_gte_lower & loc_lt_upper

            bin_val = BinResultValue(cut_range.lower_cut_point_id, lower_cut_point_value,
                                     cut_range.upper_cut_point_id, upper_cut_point_value)

            result_item_map[col] = ResultItem(col, col_info.column_type, bin_val, loc)

        result_item_list: list[ResultItem] = []
        for col in index.get_columns_after(None):
            if col in result_item_map:
                result_item_list.append(result_item_map[col])
        result_path: ResultPath = ResultPath(result_item_list, self.locations, self.tree.search_mode)
        return result_path
