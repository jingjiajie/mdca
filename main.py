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
    #                                                               affect_threshold_ratio=0.05)

    # data_df: pd.DataFrame = pd.read_csv('data/hmeq/hmeq_train.csv')
    # data_df: pd.DataFrame = pd.read_csv('data/flights/flights_processed.csv')

    # data_df.dropna(inplace=True, subset=['AIR_SYSTEM_DELAY'])

    data_df: pd.DataFrame = pd.read_csv('data/tianchi-loan/pred_2011.csv')

    analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, target_column='isError',
                                                                  target_value='1', is_sas_dataset=False,
                                                                  affect_threshold_ratio=0.02)

    # results = [ResultPath(items=[ResultItem('isError', '1'), ResultItem('verificationStatus', '1')]),
    #            ResultPath(items=[ResultItem('isError', '1'), ResultItem('term', '5')]),
    #            ResultPath(items=[ResultItem('isError', '1'), ResultItem('term', '5'), ResultItem('verificationStatus', '1')])]
    # results = chi2_filter(results, analyzer.target_column, analyzer._full_index)

    # results = [ResultPath(items=[ResultItem('BAD', '1'), ResultItem('REASON', 'JonasReason')]),
    #            ResultPath(items=[ResultItem('BAD', '1'), ResultItem('JOB', 'JonasJob')]),
    #            ResultPath(items=[ResultItem('BAD', '1'), ResultItem('REASON', 'JonasReason'), ResultItem('JOB', 'JonasJob')])]
    # results = chi2_filter(results, analyzer.target_column, analyzer._full_index)

    results: list[ResultPath] = analyzer.run()
    for r in results:
        loc: np.ndarray | None = None
        for item in r.items:
            cur_loc: np.ndarray = analyzer._full_index.get_locations(item.column, item.value)
            if loc is None:
                loc = cur_loc
            else:
                loc = loc & cur_loc
        target_affect_count = (loc & analyzer._full_index.get_locations(analyzer.target_column, analyzer.target_value)).sum()
        ratio_target: float = target_affect_count / len(analyzer._processed_target_df)
        ratio_full: float = loc.sum() / len(analyzer._processed_full_data_df)
        ratio_raise = ratio_target / ratio_full
        if ratio_target >= 0.1 and ratio_raise >= 1.5:
            print(r, "COUNT:%.2f" % target_affect_count, ", RATIO_FULL: %.4f" % ratio_full, ", RATIO_TARGET: %.4f" % ratio_target, 'RAISE: %.2f' % ratio_raise)
