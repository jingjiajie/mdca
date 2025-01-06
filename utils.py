from typing import Any, Callable


def bin_search_nearest_lower_int(search_list_asc: list[int], value: int, low: int | None = None, high: int | None = None) -> int:
    """
    :param high: search end index
    :param low: search start index
    :param search_list_asc: list to search
    :param value: value to search
    :return: -1 if value is the lowest, otherwise the index of nearest lower element
    """
    if len(search_list_asc) == 0:
        return -1
    if low is None or low < 0:
        low = 0
    if high is None or high >= len(search_list_asc):
        high = len(search_list_asc) - 1
    while low < high:
        mid: int = int((low + high) / 2)
        cur: int = search_list_asc[mid]
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
    if search_list_asc[low] > value:
        low -= 1
    return low


# def bin_search_nearest_lower(search_list: list[Any], get_value: Callable[[Any], int],
#                              value: int, low: int | None = None, high: int | None = None) -> int:
#     """
#     :param get_value: get int value from items of search_list
#     :param high: search end index
#     :param low: search start index
#     :param search_list: list to search
#     :param value: value to search
#     :return: -1 if value is the lowest, otherwise the index of nearest lower element
#     """
#     search_list_int: list[int] = [] * len(search_list)
#     i: int = 0
#     for item in search_list:
#         val: int = get_value(item)
#         search_list_int[i] = val
#         i += 1
#     return bin_search_nearest_lower_int(search_list_int, value, low, high)


