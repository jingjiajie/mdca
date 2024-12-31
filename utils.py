def bin_search_nearest_lower(search_list: list[int], value: int, low: int | None = None, high: int | None = None) -> int:
    """
    :param high: search end index
    :param low: search start index
    :param search_list: list to search
    :param value: value to search
    :return: -1 if value is the lowest, otherwise the index of nearest lower element
    """
    if len(search_list) == 0:
        return -1
    if low is None or low < 0:
        low = 0
    if high is None or high >= len(search_list):
        high = len(search_list) - 1
    while low < high:
        mid: int = int((low + high) / 2)
        cur: int = search_list[mid]
        if cur < value:
            if mid == low:
                low += 1
            else:
                low = mid
        elif cur > value:
            if mid == high:
                high -= 1
            else:
                high = mid
        else:  # cur == rand
            return mid
    if search_list[low] > value:
        low -= 1
    return low