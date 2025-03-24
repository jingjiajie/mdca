import time

import numpy as np
import pandas as pd

from mdca.analyzer.BinMerger import BinMerger
from mdca.analyzer.Index import Index
from mdca.analyzer.MCTSTree import MCTSTree
from mdca.analyzer.ResultCluster import ResultClusterSet
from mdca.analyzer.ResultPath import ResultPath, CalculatedResult
from mdca.analyzer.DataPreprocessor import DataPreprocessor, ProcessResult
from mdca.analyzer.chi2_filter import chi2_filter, chi2_filter_distribution
from mdca.analyzer.commons import Value, ColumnInfo

BIN_NUMBER: int = 50
MIN_BIN: int = 1

SORT_UNIQUE_VALUES_THRESHOLD = 20


class MultiDimensionalAnalyzer:

    def __init__(self,
                 data_df: pd.DataFrame,
                 search_mode: str,
                 columns: list[str] | None = None,
                 target_column: str | None = None,
                 prediction_column: str | None = None,
                 target_value: str | None = None,
                 min_coverage: float | None = None,
                 min_target_coverage: float | None = None,
                 min_error_coverage: float | None = None):
        if search_mode == 'error':
            if target_column is None:
                raise Exception('target_column must be specified for search_mode: error')
            elif prediction_column is None:
                raise Exception('prediction_column must be specified for search_mode: error')
            elif target_value is not None:
                raise Exception('target_value should not be specified for search_mode: error')
        elif search_mode == 'fairness':
            if target_column is None:
                raise Exception('target_column must be specified for search_mode: fairness')
            elif prediction_column is not None:
                raise Exception('prediction_column is not yet supported for search_mode: fairness')
            elif target_value is None:
                raise Exception('target_value must be specified for search_mode: fairness')
        elif search_mode == 'distribution':
            if prediction_column is not None:
                raise Exception('prediction_column should not be specified for search_mode: distribution')
        else:
            raise Exception('search_mode must be fairness, distribution or error, actual: %s' % search_mode)

        if columns is None:
            columns = []
            for col in data_df.columns:
                col: str
                if col.startswith('Unnamed:'):
                    continue
                columns.append(col)

        if target_column is not None and target_column not in columns:
            columns.append(target_column)
        if prediction_column is not None and prediction_column not in columns:
            columns.append(prediction_column)

        for col in columns:
            if col not in data_df.columns:
                raise Exception('Column (%s) not exist in data table. Existing columns: [%s]' %
                                (col, ', '.join(data_df.columns)))

        drop_cols: list[str] = []
        for col in data_df.columns:
            if col not in columns:
                drop_cols.append(col)
        data_df.drop(drop_cols, axis=1, inplace=True)

        if search_mode == 'error':
            data_df['__isError__'] = data_df[target_column] != data_df[prediction_column]
            data_df.drop([prediction_column, target_column], axis=1, inplace=True)
            target_column = '__isError__'
            target_value = 'True'
            min_target_coverage = min_error_coverage

        self.min_coverage: float | None = min_coverage
        self.min_target_coverage: float | None = min_target_coverage
        self.search_mode: str = search_mode

        preprocessor: DataPreprocessor = DataPreprocessor()
        process_result: ProcessResult = preprocessor.process(data_df, target_column, min_coverage)
        self.column_info: dict[str, ColumnInfo] = process_result.column_info
        self.processed_data_df: pd.DataFrame = process_result.data_df

        self.target_column: str | None = target_column  # Target column is never binned
        self.target_value: Value | None
        if target_value is None:
            self.target_value = None
        else:
            self._init_target_value(target_value, target_column, self.column_info[target_column].column_type)
        if target_column is not None:
            if not np.any(data_df[target_column].unique() == self.target_value):
                raise Exception('Target value (%s) can not be found in target column: (%s), possible values: %s' %
                                (self.target_value, self.target_column,
                                 '[' + ', '.join(data_df[target_column].unique().astype(str)) + ']'))

        data_index: Index = Index(self.processed_data_df, self.target_column, self.target_value, self.column_info)
        self.data_index = data_index

    def _init_target_value(self, target_value: str | None, target_column: str, target_col_type: str) -> None:
        if target_value is None:
            self.target_value = None
            return
        target_value = target_value.strip()

        def is_number(s: str) -> bool:
            try:
                float(s)
                return True
            except ValueError:
                return False

        if target_col_type == 'bool':
            if target_value in ['0', '1']:
                self.target_value = target_value == '1'
            elif target_value.lower() in ['true', 'false']:
                self.target_value = target_value.lower() == 'true'
            elif target_value.lower() in ['null', 'none', 'nan', '']:
                self.target_value = None
            else:
                raise Exception('Can not convert target value (%s) to target column (%s) type: %s' %
                                (target_value, target_column, target_col_type))
        elif target_col_type == 'int':
            if is_number(target_value) and int(target_value) == float(target_value):
                self.target_value = int(target_value)
            elif target_value.lower() in ['null', 'none', 'nan', '']:
                self.target_value = np.nan
            else:
                raise Exception('Can not convert target value (%s) to target column (%s) type: %s' %
                                (target_value, target_column, target_col_type))
        elif target_col_type == 'float':
            if is_number(target_value):
                self.target_value = float(target_value)
            elif target_value.lower() in ['null', 'none', 'nan', '']:
                self.target_value = np.nan
            else:
                raise Exception('Can not convert target value (%s) to target column (%s) type: %s' %
                                (target_value, target_column, target_col_type))
        elif target_col_type == 'str':
            if target_value.lower() in ['']:
                self.target_value = None
            else:
                self.target_value = str(target_value)
        else:
            raise Exception('Unexpected type (%s) of target column (%s)' % (target_col_type, target_column))

    def run(self, mcts_rounds: int = 100000, max_results: int = 20) -> list[CalculatedResult]:
        tree: MCTSTree | None = MCTSTree(self.data_index, self.column_info, self.target_column, self.target_value,
                                         self.search_mode, self.min_coverage, self.min_target_coverage)
        tree.run(mcts_rounds)
        print('Filtering results...')
        chi2_cost: float = 0
        cluster_cost: float = 0
        result_cluster_set_inc: ResultClusterSet = ResultClusterSet()
        result_cluster_set_dec: ResultClusterSet = ResultClusterSet()
        existing_result_set: set[str] = set()
        while len(result_cluster_set_inc) < max_results or len(result_cluster_set_dec) < max_results:
            result: ResultPath | None = tree.next_result()
            if result is None:
                break
            calculated_res: CalculatedResult = result.calculate(self.data_index)

            start_time: float = time.time()
            if self.search_mode in ['fairness', 'error']:
                calculated_res = chi2_filter(calculated_res, self.data_index, self.search_mode)
            elif self.search_mode == 'distribution':
                calculated_res = chi2_filter_distribution(calculated_res, self.data_index)
            chi2_cost += time.time() - start_time

            if calculated_res is None:
                continue
            elif str(calculated_res) in existing_result_set:
                continue
            else:
                existing_result_set.add(str(calculated_res))

            start_time = time.time()
            if self.search_mode in ['fairness', 'error']:
                if calculated_res.target_rate >= self.data_index.total_target_rate:
                    if len(result_cluster_set_inc) >= max_results:
                        continue
                    result_cluster_set_inc.cluster_result(calculated_res)
                else:
                    if len(result_cluster_set_dec) >= max_results:
                        continue
                    result_cluster_set_dec.cluster_result(calculated_res)
            elif self.search_mode == 'distribution':
                if calculated_res.coverage >= calculated_res.baseline_coverage:
                    if len(result_cluster_set_inc) >= max_results:
                        continue
                    result_cluster_set_inc.cluster_result(calculated_res)
                else:
                    if len(result_cluster_set_dec) >= max_results:
                        continue
                    result_cluster_set_dec.cluster_result(calculated_res)
            cluster_cost += time.time() - start_time
        results: list[ResultPath] = result_cluster_set_inc.get_results() + result_cluster_set_dec.get_results()
        del tree
        print("Chi2 test cost: %.2f seconds" % chi2_cost)
        print("Clustering results cost: %.2f seconds" % cluster_cost)

        merger: BinMerger = BinMerger(self.data_index, self.column_info, self.search_mode)
        results = merger.expand(results)
        results = merger.merge(results)
        results = merger.filter(results)

        # remove duplicated results
        result_map: dict[str, ResultPath] = {}
        for res in results:
            if len(res.items) == 0:
                continue
            if str(res) not in result_map:
                result_map[str(res)] = res
        results = list(result_map.values())
        calculated_results: list[CalculatedResult] = list(map(lambda r: r.calculate(self.data_index), results))
        calculated_results = sorted(calculated_results, key=lambda r: r.weight, reverse=True)
        return calculated_results

    def print_results(self, results: list[CalculatedResult]):
        index: Index = self.data_index
        if self.search_mode == 'fairness':
            print('\n========== Overall ============')
            print("Total rows: %d" % index.total_count)
            print("Overall target rate: %5.2f%%" % (index.total_target_rate * 100))

            def _print_fairness_results(results: list[CalculatedResult]):
                print('Coverage(Count),\tTarget Rate(Overall+N%),\tResult')
                for r in results:
                    count: int = r.count
                    target_rate: float = r.target_rate
                    coverage: float = r.coverage
                    print("%5.2f%% (%6d),\t%5.2f%% (%+6.2f%%),\t\t%s" %
                          (100 * coverage,
                           count,
                           100 * target_rate,
                           100 * (target_rate - index.total_target_rate),
                           str(r))
                          )

            print('\n========== Results of Target Rate Increase ============')
            res_inc: list[CalculatedResult] = list(
                filter(lambda r: (r.target_rate >= index.total_target_rate), results))
            res_inc = sorted(res_inc, key=lambda r: r.weight, reverse=True)
            _print_fairness_results(res_inc)
            print('\n========== Results of Target Rate Decrease ============')
            res_dec: list[CalculatedResult] = list(filter(lambda r: (r.target_rate < index.total_target_rate), results))
            res_dec = sorted(res_dec, key=lambda r: r.weight, reverse=True)
            _print_fairness_results(res_dec)

        elif self.search_mode == 'error':
            print('\n========== Overall ============')
            print("Total rows: %d" % index.total_count)
            print("Overall error rate: %5.2f%%" % (index.total_target_rate * 100))

            def _print_fairness_results(results: list[CalculatedResult]):
                print('Error Coverage(Count),\tError Rate(Overall+N%),\tResult')
                for r in results:
                    target_count: int = r.target_count
                    target_rate: float = r.target_rate
                    target_coverage: float = r.target_coverage
                    print("%5.2f%% (%6d),\t\t%5.2f%% (%+6.2f%%),\t\t%s" %
                          (100 * target_coverage,
                           target_count,
                           100 * target_rate,
                           100 * (target_rate - index.total_target_rate),
                           str(r))
                          )

            print('\n========== Results of Error Rate Increase ============')
            res_inc: list[CalculatedResult] = list(
                filter(lambda r: (r.target_rate >= index.total_target_rate), results))
            res_inc = sorted(res_inc, key=lambda r: r.weight, reverse=True)
            _print_fairness_results(res_inc)
            print('\n========== Results of Error Rate Decrease ============')
            res_dec: list[CalculatedResult] = list(filter(lambda r: (r.target_rate < index.total_target_rate), results))
            res_dec = sorted(res_dec, key=lambda r: r.weight, reverse=True)
            _print_fairness_results(res_dec)

        elif self.search_mode == 'distribution':
            print('\n========== Overall ============')
            print("Total rows: %d" % index.total_count)
            if self.target_column is not None:
                print("Overall target rate: %5.2f%%" % (index.total_target_rate * 100))

            def _print_distribution_results(results: list[CalculatedResult]):
                if self.target_column is not None:
                    print('Coverage (Baseline, +N%, *X),\t\tTarget Rate(Overall +%N),\tResult')
                    for res in results:
                        coverage: float = res.coverage
                        baseline_coverage: float = res.baseline_coverage
                        target_rate: float = res.target_rate
                        print("%5.2f%% (%5.2f%%, %+6.2f%%, *%-5.2f),\t%5.2f%% (%+6.2f%%),\t\t%s" %
                              (100 * coverage,
                               100 * baseline_coverage,
                               100 * (coverage - baseline_coverage),
                               (coverage / baseline_coverage),
                               100 * target_rate,
                               100 * (target_rate - index.total_target_rate),
                               str(res))
                              )
                else:
                    print('Coverage (Baseline, +N%, *X),\t\tResult')
                    for res in results:
                        coverage: float = res.coverage
                        baseline_coverage: float = res.baseline_coverage
                        print("%5.2f%% (%5.2f%%, %+6.2f%%, *%-5.2f),\t%s" %
                              (100 * coverage,
                               100 * baseline_coverage,
                               100 * (coverage - baseline_coverage),
                               (coverage / baseline_coverage),
                               str(res)))

            print('\n========== Results of Coverage Increase ============')
            res_inc: list[CalculatedResult] = list(filter(lambda r: (r.coverage >= r.baseline_coverage), results))
            res_inc = sorted(res_inc, key=lambda r: r.weight, reverse=True)
            _print_distribution_results(res_inc)

            print('\n========== Results of Coverage Decrease ============')
            res_dec: list[CalculatedResult] = list(filter(lambda r: (r.coverage < r.baseline_coverage), results))
            res_dec = sorted(res_dec, key=lambda r: r.weight, reverse=True)
            _print_distribution_results(res_dec)
