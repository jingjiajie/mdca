class ResultItem:

    def __init__(self, column: str, value: str):
        self.column: str = column
        self.value: str = value


class ResultPath:

    def __init__(self, items: list[ResultItem]):
        self.items: list[ResultItem] = items