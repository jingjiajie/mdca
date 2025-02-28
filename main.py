import random
import time
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from analyzer.Index import Index
from analyzer.MultiDimensionalAnalyzer import MultiDimensionalAnalyzer
from analyzer.ResultPath import ResultPath, CalculatedResult


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
    # search_mode: str = 'fairness'
    search_mode: str = 'distribution'

    start: float = time.time()
    # data_df: pd.DataFrame = mock_hmeq_data()
    #
    # analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, target_column='BAD', target_value=1,
    #                                                               min_coverage=0.05, search_mode=search_mode)
    # data_df: pd.DataFrame = pl.read_csv('data/hmeq/hmeq_train.csv').to_pandas()
    #
    # analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, target_column='BAD', target_value=1,
                                                                    # min_coverage=0.05, search_mode=search_mode)

    data_df = pl.read_csv('data/flights/flights.csv', encoding="utf8-lossy").to_pandas()

    data_df['DELAYED'] = ~(data_df['AIR_SYSTEM_DELAY'].isna() & data_df['SECURITY_DELAY'].isna() &
                             data_df['AIRLINE_DELAY'].isna() & data_df['LATE_AIRCRAFT_DELAY'].isna() &
                             data_df['WEATHER_DELAY'].isna())
    data_df.drop(['DEPARTURE_DELAY','ARRIVAL_DELAY','AIR_SYSTEM_DELAY', 'SECURITY_DELAY',
                  'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'],
                 axis=1, inplace=True)

    data_df = data_df[['YEAR','MONTH','DAY','DAY_OF_WEEK','AIRLINE','FLIGHT_NUMBER',
                       'TAIL_NUMBER','ORIGIN_AIRPORT','DESTINATION_AIRPORT','DELAYED']]
    analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, target_column='DELAYED', target_value=1,
                                                                  min_coverage=0.05, search_mode=search_mode)

    # data_df: pd.DataFrame = pd.read_csv('data/tianchi-loan/pred_2011.csv')
    # data_df = data_df[data_df['term'] != 6]
    # analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, target_column='isError', target_value=1,
    #                                                               min_coverage=0.02, search_mode=search_mode)

    # print('Loading data...')
    # data_df: pd.DataFrame = pl.read_csv('data/recruitment/recruitmentdataset-2022-1.3.csv', encoding="utf8-lossy").to_pandas()
    # print('Load data cost: %.2f seconds' % (time.time() - start))
    # analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, target_column='decision', target_value=True,
    #                                                               search_mode=search_mode, min_coverage=0.05)
    # data_df.drop(['decision'], axis=1, inplace=True)
    # analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, None, None,
    #                                                               search_mode=search_mode, min_coverage=0.05)

    results: list[CalculatedResult] = analyzer.run()
    index: Index = analyzer.data_index
    print("\nTotal time cost: %.2f seconds" % (time.time() - start))

    if search_mode == 'fairness':
        print('\n========== Overall ============')
        print("Total rows: %d" % index.total_count)
        print("Baseline target rate: %5.2f%%" % (index.total_target_rate * 100))

        def _print_fairness_results(results: list[CalculatedResult]):
            print('Target Rate(Baseline+N%),\tTarget Coverage(Count),\tResult')
            for r in results:
                target_count: int = r.target_count
                target_rate: float = r.target_rate
                target_coverage: float = r.target_coverage
                print("%5.2f%% (%+6.2f%%),\t\t\t%5.2f%% (%6d),\t%s" %
                      (100 * target_rate,
                       100 * (target_rate - index.total_target_rate),
                       100 * target_coverage,
                       target_count,
                       str(r))
                      )

        print('\n========== Results of Target Rate Increase ============')
        res_inc: list[CalculatedResult] = filter(lambda r: (r.target_rate >= index.total_target_rate), results)
        res_inc = sorted(res_inc, key=lambda r: r.weight, reverse=True)
        _print_fairness_results(res_inc)
        print('\n========== Results of Target Rate Decrease ============')
        res_dec: list[CalculatedResult] = filter(lambda r: (r.target_rate < index.total_target_rate), results)
        res_dec = sorted(res_dec, key=lambda r: r.weight, reverse=True)
        _print_fairness_results(res_dec)

    elif search_mode == 'distribution':
        print('\n========== Overall ============')
        print("Total rows: %d" % index.total_count)

        def _print_distribution_results(results: list[CalculatedResult]):
            print('Coverage,\tBaseline,\tBaseline+N%,\tBaseline*X,\tResult')
            for res in results:
                coverage: float = res.total_coverage
                baseline_coverage: float = res.baseline_coverage
                print("%5.2f%%,\t\t%5.2f%%,\t\t%+6.2f%%,\t\t*%-5.2f,\t\t%s, %.2f" %
                      (100 * coverage,
                       100 * baseline_coverage,
                       100 * (coverage - baseline_coverage),
                       (coverage / baseline_coverage),
                       str(res), res.weight)
                      )

        print('\n========== Results of Coverage Increase ============')
        res_inc: list[CalculatedResult] = filter(lambda r: (r.total_coverage >= r.baseline_coverage), results)
        res_inc = sorted(res_inc, key=lambda r: r.weight, reverse=True)
        _print_distribution_results(res_inc)

        print('\n========== Results of Coverage Decrease ============')
        res_dec: list[CalculatedResult] = filter(lambda r: (r.total_coverage < r.baseline_coverage), results)
        res_dec = sorted(res_dec, key=lambda r: r.weight, reverse=True)
        _print_distribution_results(res_dec)
