import math

import numpy as np
import pandas as pd

data_df: pd.DataFrame = pd.read_csv('data/hmeq/hmeq_train_converted.csv')
# hmeq_train_df = hmeq_train_df.map(lambda item: np.nan if str(item).strip() == '.' else item)
# hmeq_train_df.to_csv('data/hmeq/hmeq_train_converted.csv')

BIN_NUMBER = 20

# Binning
for col_name in data_df.columns:
    if data_df.dtypes[col_name].name == 'float64':
        min_int: int = math.floor(data_df[col_name].min())
        max_int: int = math.ceil(data_df[col_name].max())
        step: float = (max_int - min_int) / BIN_NUMBER
        if step < 1:
            step = 1
        bins: list[int] = []
        cur_bin: float = min_int
        while cur_bin <= max_int:
            if math.ceil(cur_bin) >= max_int:
                bins.append(max_int)
            else:
                bins.append(math.floor(cur_bin))
            cur_bin += step
        data_df[col_name] = pd.cut(data_df[col_name], bins=bins, include_lowest=True)

print(data_df)

