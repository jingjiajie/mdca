import random
import time
from typing import Any

import numpy as np
import pandas as pd

from analyzer.MultiDimensionalAnalyzer import MultiDimensionalAnalyzer
from analyzer.ResultPath import ResultPath, ResultItem
from analyzer.chi2_filter import chi2_filter


def mock_hmeq_data() -> pd.DataFrame:
    data_df: pd.DataFrame = pd.read_csv('data/hmeq/hmeq_train.csv')
    unique_val_map: dict[str, dict[Any, float]] = {}  # col -> val -> prob
    for col in data_df.columns:
        unique_vals: pd.Series = data_df[col].unique()
        unique_val_map[col] = {}
        for val in unique_vals:
            val: Any
            freq: int
            if (type(val) is float or type(val) is np.float64) and np.isnan(val):
                freq = len(data_df[data_df[col].isna()])
                unique_val_map[col]['nan'] = freq / len(data_df)
            else:
                freq = len(data_df[data_df[col] == val])
                unique_val_map[col][val] = freq / len(data_df)

    new_rows = []

    for i in range(0, 200):
        new_row = {}
        for c in data_df.columns:
            if c == 'JOB':
                new_row[c] = 'JonasJob'
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

    for i in range(0, 200):
        new_row = {}
        for c in data_df.columns:
            if c == 'REASON':
                new_row[c] = 'JonasReason'
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

    for i in range(0, 200):
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

    # data_df: pd.DataFrame = mock_hmeq_data()

    # analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, target_column='BAD',
    #                                                               target_value='1', is_sas_dataset=True,
    #                                                               min_error_coverage=0.2)

    # data_df: pd.DataFrame = pd.read_csv('data/hmeq/hmeq_train.csv')
    # data_df: pd.DataFrame = pd.read_csv('data/flights/flights_processed.csv')

    # data_df.dropna(inplace=True, subset=['AIR_SYSTEM_DELAY'])

    data_df: pd.DataFrame = pd.read_csv('data/tianchi-loan/pred_2011.csv')

    analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, target_column='isError',
                                                                  target_value='1', is_sas_dataset=False,
                                                                  min_error_coverage=0.05)

    # results = [ResultPath(items=[ResultItem('isError', '1'), ResultItem('verificationStatus', '1')]),
    #            ResultPath(items=[ResultItem('isError', '1'), ResultItem('term', '5')]),
    #            ResultPath(items=[ResultItem('isError', '1'), ResultItem('term', '5'), ResultItem('verificationStatus', '1')])]
    # results = chi2_filter(results, analyzer.target_column, analyzer._full_index)

    # results = [ResultPath(items=[ResultItem('BAD', '1'), ResultItem('REASON', 'JonasReason')]),
    #            ResultPath(items=[ResultItem('BAD', '1'), ResultItem('JOB', 'JonasJob')]),
    #            ResultPath(items=[ResultItem('BAD', '1'), ResultItem('REASON', 'JonasReason'), ResultItem('JOB', 'JonasJob')])]
    # results = chi2_filter(results, analyzer.target_column, analyzer._full_index)

    results: list[ResultPath] = analyzer.run()
    all_error_loc: pd.Series = analyzer._full_index.get_locations(analyzer.target_column, analyzer.target_value)
    for r in results:
        loc: np.ndarray | None = None
        for item in r.items:
            cur_loc: np.ndarray = analyzer._full_index.get_locations(item.column, item.value)
            if loc is None:
                loc = cur_loc
            else:
                loc = loc & cur_loc
        error_count = (loc & all_error_loc).sum()
        error_rate: float = error_count / loc.sum()
        error_coverage: float = error_count / all_error_loc.sum()
        print(r, '\t', "error_count: %d" % error_count, ", error_coverage: %.2f" % error_coverage, ", error_rate: %.2f" % error_rate)
