import random
import time
from typing import Any

import numpy as np
import pandas as pd

from analyzer.MultiDimensionalAnalyzer import MultiDimensionalAnalyzer
from analyzer.ResultPath import ResultPath


def mock_hmeq_data() -> pd.DataFrame:
    data_df: pd.DataFrame = pd.read_csv('data/hmeq/hmeq_train.csv')
    unique_val_map: dict[str, dict[Any, float]] = {}  # col -> val -> prob
    for col in data_df.columns:
        unique_vals: pd.Series = data_df[col].unique()
        unique_val_map[col] = {}
        for val in unique_vals:
            val: Any
            freq: int = len(data_df[data_df[col] == val])
            unique_val_map[col][val] = freq / len(data_df)

    new_rows = []
    for i in range(0, 1000):
        new_row = {}
        for c in data_df.columns:
            if c == 'JOB':
                new_row[c] = 'JonasJob'
            elif c == 'REASON':
                new_row[c] = 'JonasReason'
            elif c == 'BAD':
                new_row[c] = 1
            else:
                unique_val_prob: dict[Any, float] = unique_val_map[c]
                unique_vals: list[Any] = []
                probs: list[float] = []
                for val in unique_val_prob:
                    unique_vals.append(val)
                    probs.append(unique_val_prob[val])
                new_val = np.random.choice(unique_vals, size=1, p=probs)[0]
                new_row[c] = new_val
        new_rows.append(new_row)
    data_df = data_df._append(new_rows, ignore_index=True)
    return data_df

if __name__ == '__main__':
    random.seed(time.time())

    data_df: pd.DataFrame = mock_hmeq_data()
    # data_df.to_csv('data/hmeq/hmeq_train_mock.csv')
    # exit(0)
    # data_df: pd.DataFrame = pd.read_csv('data/hmeq/hmeq_train.csv')
    # data_df: pd.DataFrame = pd.read_csv('data/flights/flights.csv')

    # data_df.dropna(inplace=True, subset=['AIR_SYSTEM_DELAY'])

    # data_df = data_df[data_df['BAD'] == 1].copy()

    analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, affect_threshold_ratio=0.05)
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
