import random
import time

import numpy as np
import pandas as pd

from analyzer.MultiDimensionalAnalyzer import MultiDimensionalAnalyzer
from analyzer.ResultPath import ResultPath

def mock_hmeq_data() -> pd.DataFrame:
    data_df: pd.DataFrame = pd.read_csv('data/hmeq/hmeq_train.csv')
    unique_val_map = {}
    for c in data_df.columns:
        unique_val_map[c] = data_df[c].unique()

    new_rows = []
    for i in range(0, 10000):
        new_row = {}
        for c in data_df.columns:
            if c == 'JOB':
                new_row[c] = 'JonasJob'
            elif c == 'REASON':
                new_row[c] = 'JonasReason'
            else:
                unique_vals: pd.Series = unique_val_map[c]
                new_val = unique_vals[random.randint(0, len(unique_vals)-1)]
                new_row[c] = new_val
        new_rows.append(new_row)
    data_df = data_df._append(new_rows, ignore_index=True)
    return data_df

if __name__ == '__main__':
    random.seed(time.time())

    # data_df: pd.DataFrame = pd.read_csv('data/hmeq/hmeq_train.csv')
    # data_df: pd.DataFrame = pd.read_csv('data/flights/flights.csv')
    data_df: pd.DataFrame = mock_hmeq_data()

    # data_df.dropna(inplace=True, subset=['AIR_SYSTEM_DELAY'])

    analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, "BAD")
    results: list[ResultPath] = analyzer.run()
    for r in results:
        loc: np.ndarray | None = None
        for item in r.items:
            if loc is None:
                loc = analyzer._index.get_locations(item.column, item.value)
            else:
                loc = loc & analyzer._index.get_locations(item.column, item.value)
        if loc is None:
            print(r, 0)
        else:
            print(r, loc.sum())
