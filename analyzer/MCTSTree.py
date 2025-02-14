from analyzer.Index import Index
from analyzer.MCTSTreeNode import MCTSTreeNode
from analyzer.ResultPath import ResultPath, ResultItem

# TODO 动态自适应
SIMULATE_TIMES: int = 10


class MCTSTree:

    def __init__(self, data_index: Index, min_error_coverage: float):
        self.data_index: Index = data_index
        self.min_error_coverage: float = min_error_coverage
        self._root: MCTSTreeNode = MCTSTreeNode(self, None, None, None)
        self._result: list[ResultPath] = []  # descending ordered

    def run(self, times: int):
        for i in range(0, times):
            selected_leaf: MCTSTreeNode = self._root.select()
            if selected_leaf.children is None:
                selected_leaf.expand()
                for child in selected_leaf.children:
                    child.simulate(SIMULATE_TIMES)
                    child.back_propagate()
            elif selected_leaf.is_root:
                break
            else:
                raise Exception('Unexpected error: MCTS selection of ' + str(selected_leaf))
        results: list[ResultPath] = self._choose_results()
        return results

    def _choose_results(self, max_results: int = 1000) -> list[ResultPath]:
        results: list[ResultPath] = []
        for i in range(0, max_results):
            cur: MCTSTreeNode = self._root
            while cur.children is not None and len(cur.children) > 0:
                max_q_child: MCTSTreeNode = cur.children[0]
                for child in cur.children:
                    if child.q_value > max_q_child.q_value:
                        max_q_child = child
                if max_q_child.q_value < cur.q_value:
                    break
                else:
                    cur = max_q_child
            if cur.is_root:
                break
            cur.pick()
            result_items: list[ResultItem] = []
            while cur.parent is not None:
                result_items.append(ResultItem(cur.column, cur.value, self.data_index.get_locations(cur.column, cur.value)))
                cur = cur.parent
            result_items.reverse()
            result_path: ResultPath = ResultPath(result_items)
            results.append(result_path)
        filtered_results: list[ResultPath] = self._filter_suffix_result(results)
        return filtered_results

    def _filter_suffix_result(self, results: list[ResultPath]) -> list[ResultPath]:
        filtered_results: list[ResultPath] = []
        for i in range(0, len(results)):
            res1: ResultPath = results[i]
            should_filter_out: bool = False
            for j in range(0, len(results)):
                if i == j:
                    continue
                res2: ResultPath = results[j]
                if len(res1.items) >= len(res2.items):
                    continue
                is_suffix: bool = True
                # Check suffix
                for k in range(0, len(res1.items)):
                    item1: ResultItem = res1.items[-k-1]
                    item2: ResultItem = res2.items[-k-1]
                    if item1 != item2:
                        is_suffix = False
                        break
                if not is_suffix:
                    continue
                if res1.locations.sum() != res2.locations.sum():
                    continue
                should_filter_out = True
            if not should_filter_out:
                filtered_results.append(res1)
        return filtered_results

