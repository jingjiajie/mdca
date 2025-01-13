from analyzer.Index import Index
from analyzer.MCTSTreeNode import MCTSTreeNode
from analyzer.ResultPath import ResultPath, ResultItem

# TODO 动态自适应
SIMULATE_TIMES: int = 10


class MCTSTree:

    def __init__(self, data_index: Index, threshold: int):
        self.data_index: Index = data_index
        self.threshold: int = threshold
        self._root: MCTSTreeNode = MCTSTreeNode(self, None, None, None)
        self._result: list[ResultPath] = []  # descending ordered

    def run(self, times: int):
        self._result = []
        for i in range(0, times):
            selected_leaf: MCTSTreeNode = self._root.select()
            if selected_leaf.children is None:
                selected_leaf.expand()

            if len(selected_leaf.children) > 0:
                for child in selected_leaf.children:
                    child.simulate(SIMULATE_TIMES)
                    child.back_propagate()
            elif selected_leaf.is_root:
                break
            else:  # len(selected_leaf.children) == 0 and not selected_leaf.is_root
                selected_leaf.pick()
                if selected_leaf.check():
                    self._add_result(selected_leaf)

        return self._result

    def _add_result(self, node: MCTSTreeNode):
        result_items: list[ResultItem] = []
        cur: MCTSTreeNode = node
        while cur.parent is not None:
            result_items.append(ResultItem(cur.column, cur.value))
            cur = cur.parent
        result_items.reverse()
        result_path: ResultPath = ResultPath(result_items)
        if len(self._result) < 200:
            self._result.append(result_path)
        elif result_path.depth > self._result[len(self._result) - 1].depth:
            self._result[len(self._result)-1] = result_path
        self._result.sort(key=lambda item: item.depth, reverse=True)

