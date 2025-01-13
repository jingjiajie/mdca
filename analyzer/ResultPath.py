class ResultItem:

    def __init__(self, column: str, value: str):
        self.column: str = column
        self.value: str = value

    def __str__(self):
        return f"{self.column}={self.value}"


class ResultPath:

    def __init__(self, items: list[ResultItem]):
        self.items: list[ResultItem] = items

    @property
    def depth(self):
        return len(self.items)

    def path(self):
        item_str_list: list[str] = []
        for item in self.items:
            item_str_list.append(str(item))
        return "[" + ", ".join(item_str_list) + "]"

    def __str__(self):
        return self.path()
