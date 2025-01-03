from Index import Index
from MCTSTreeNode import MCTSTreeNode


class MCTSTree:

    def __init__(self, data_index: Index, threshold: int):
        self.data_index: Index = data_index
        self.threshold: int = threshold
        self._root: MCTSTreeNode = MCTSTreeNode(self, None, None, None)
        self._result: list[MCTSTreeNode] = []  # descending ordered

    def run(self, times: int):
        self._result = []
        for i in range(0, times):
            selected_leaf: MCTSTreeNode = self._root.select()
            if selected_leaf.children is None:
                selected_leaf.expand()

            if len(selected_leaf.children) > 0:
                for child in selected_leaf.children:
                    child.simulate()
                    child.back_propagate()
            elif selected_leaf.is_root:
                break
            else:  # len(selected_leaf.children) == 0 and not selected_leaf.is_root
                selected_leaf.pick()
                if selected_leaf.check():
                    self._add_result(selected_leaf)

        return self._result

    def _add_result(self, node: MCTSTreeNode):
        if len(self._result) < 10:
            self._result.append(node)
        elif node.depth > self._result[len(self._result) - 1].depth:
            self._result[len(self._result)-1] = node
        self._result.sort(key=lambda item: item.depth, reverse=True)

