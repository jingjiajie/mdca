import time

import numpy as np
import pandas as pd

from analyzer.BinMerger import BinMerger
from analyzer.Index import Index
from analyzer.MCTSTree import MCTSTree
from analyzer.ResultCluster import ResultClusterSet
from analyzer.ResultPath import ResultPath, CalculatedResult
from analyzer.DataPreprocessor import DataPreprocessor, ProcessResult
from analyzer.chi2_filter import chi2_filter
from analyzer.commons import Value, ColumnInfo

BIN_NUMBER: int = 50
MIN_BIN: int = 1

SORT_UNIQUE_VALUES_THRESHOLD = 20


class MultiDimensionalAnalyzer:

    def __init__(self, data_df: pd.DataFrame, target_column: str | None, target_value: Value | None, search_mode: str,
                 min_coverage: float = 0.05):
        if search_mode not in ['fairness', 'distribution']:
            raise Exception('search_mode must be fairness or distribution, actual: %s' % search_mode)
        self.min_coverage: float = min_coverage
        self.search_mode: str = search_mode

        preprocessor: DataPreprocessor = DataPreprocessor()
        process_result: ProcessResult = (
            preprocessor.process(data_df, target_column, target_value, min_coverage, search_mode))
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
                raise Exception('Target value %s can not be found in target column: %s, possible values: %s' %
                                (self.target_value, self.target_column,
                                 '[' + ', '.join(data_df[target_column].unique().astype(str)) + ']'))

        data_index: Index = Index(self.processed_data_df, target_column, target_value, self.column_info)
        self.data_index = data_index

    def _init_target_value(self, target_value: Value, target_column: str, target_col_type: str) -> None:
        if target_value is None:
            self.target_value = None
            return
        if isinstance(target_value, float) and np.isnan(target_value):
            self.target_value = np.nan
            return

        if target_col_type == 'bool':
            if np.issubdtype(type(target_value), bool):
                self.target_value = bool(target_value)
            elif np.issubdtype(type(target_value), int):
                self.target_value = target_value != 0
            elif np.issubdtype(type(target_value), str):
                self.target_value = target_value.strip().lower() == 'true'
            else:
                raise Exception('Can not convert target value %s to target column %s type: %s' %
                                (target_value, target_column, target_col_type))
        elif target_col_type == 'int':
            if np.issubdtype(type(target_value), int):
                self.target_value = int(target_value)
            elif np.issubdtype(type(target_value), float):
                self.target_value = int(target_value)
            elif np.issubdtype(type(target_value), str):
                self.target_value = int(target_value)
            else:
                raise Exception('Can not convert target value %s to target column %s type: %s' %
                                (target_value, target_column, target_col_type))
        elif target_col_type == 'float':
            if np.issubdtype(type(target_value), int):
                self.target_value = float(target_value)
            elif np.issubdtype(type(target_value), float):
                self.target_value = float(target_value)
            elif np.issubdtype(type(target_value), str):
                self.target_value = float(target_value)
            else:
                raise Exception('Can not convert target value %s to target column %s type: %s' %
                                (target_value, target_column, target_col_type))
        elif target_col_type == 'str':
            self.target_value = str(target_value)
        else:
            raise Exception('Unexpected type %s of target column %s' % (target_col_type, target_column))

    def run(self, mcts_rounds: int = 100000, max_results: int = 20) -> list[CalculatedResult]:
        tree: MCTSTree | None = MCTSTree(self.data_index, self.column_info, self.search_mode, self.min_coverage)
        tree.run(mcts_rounds)

        start_time: float = time.time()
        result_cluster_set_inc: ResultClusterSet = ResultClusterSet()
        result_cluster_set_dec: ResultClusterSet = ResultClusterSet()
        while len(result_cluster_set_inc) < max_results or len(result_cluster_set_dec) < max_results:
            result: ResultPath | None = tree.next_result()
            if result is None:
                break
            if self.search_mode == 'fairness':
                result = chi2_filter(result, self.search_mode)
                if result is None:
                    continue
            calculated_res: CalculatedResult = result.calculate(self.data_index)
            if self.search_mode == 'fairness':
                if calculated_res.target_rate >= self.data_index.total_target_rate:
                    if len(result_cluster_set_inc) >= max_results:
                        continue
                    result_cluster_set_inc.cluster_result(calculated_res)
                else:
                    if len(result_cluster_set_dec) >= max_results:
                        continue
                    result_cluster_set_dec.cluster_result(calculated_res)
            elif self.search_mode == 'distribution':
                if calculated_res.total_coverage >= calculated_res.baseline_coverage:
                    if len(result_cluster_set_inc) >= max_results:
                        continue
                    result_cluster_set_inc.cluster_result(calculated_res)
                else:
                    if len(result_cluster_set_dec) >= max_results:
                        continue
                    result_cluster_set_dec.cluster_result(calculated_res)
        results: list[ResultPath] = result_cluster_set_inc.get_results() + result_cluster_set_dec.get_results()
        del tree
        if self.search_mode == 'fairness':
            print("Clustering results (+Chi2-test) cost: %.2f seconds" % (time.time() - start_time))
        elif self.search_mode == 'distribution':
            print("Clustering results cost: %.2f seconds" % (time.time() - start_time))

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
        if self.search_mode == 'fairness':
            calculated_results = (
                sorted(calculated_results, key=lambda r: r.weight, reverse=True))
        elif self.search_mode == 'distribution':
            calculated_results = sorted(calculated_results,
                                        key=lambda r: (r.total_coverage > r.baseline_coverage, r.weight),
                                        reverse=True)
        return calculated_results
