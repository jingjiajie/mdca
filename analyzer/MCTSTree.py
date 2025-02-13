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
            # TODO 全部遍历完成提前停止！
            selected_leaf: MCTSTreeNode = self._root.select()
            if selected_leaf.children is None:
                selected_leaf.expand()

            if len(selected_leaf.children) > 0:
                for child in selected_leaf.children:
                    child.simulate(SIMULATE_TIMES)
                    child.back_propagate()
            elif selected_leaf.is_root:
                break
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
        return results
