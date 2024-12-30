import math

import pandas as pd

BIN_NUMBER: int = 50
MIN_BIN: int = 1


def binning_inplace(data_df: pd.DataFrame, bin_cols: list[str]):
    # Binning
    for col_name in bin_cols:
        min_int: int = math.floor(data_df[col_name].min())
        max_int: int = math.ceil(data_df[col_name].max())
        step: float = (max_int - min_int) / BIN_NUMBER
        if step < MIN_BIN:
            step = MIN_BIN
        bins: list[int] = []
        cur_bin: float = min_int
        while cur_bin <= max_int:
            if max_int - cur_bin < MIN_BIN:
                bins.append(max_int)
            else:
                bins.append(math.floor(cur_bin))
            cur_bin += step
        data_df[col_name] = pd.cut(data_df[col_name], bins=bins, include_lowest=True, right=False)