from Index import Index


class MCTSTreeNode:

    def __init__(self, parent, index: Index, column: str, value: str):
        self.parent: MCTSTreeNode = parent
        self.index: Index = index
        self.visit_count = 0
        self.column: str = column
        self.value: str = value
        self.influence_count: int
        self.children: list[MCTSTreeNode] = []

    def simulate(self):
        columns_after = self.index.get_columns_after(self.column)