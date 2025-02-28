import gc
import time

import numpy as np
import pandas as pd

from analyzer.BinMerger import BinMerger
from analyzer.Index import Index
from analyzer.MCTSTree import MCTSTree
from analyzer.ResultPath import ResultPath
from analyzer.DataPreprocessor import DataPreprocessor, ProcessResult
from analyzer.chi2_filter import chi2_filter
from analyzer.commons import Value

BIN_NUMBER: int = 50
MIN_BIN: int = 1

SORT_UNIQUE_VALUES_THRESHOLD = 20


class MultiDimensionalAnalyzer:

    def __init__(self, data_df: pd.DataFrame, target_column: str, target_value: Value,
                 min_target_coverage: float = 0.05):
        self.min_target_coverage: float = min_target_coverage

        preprocessor: DataPreprocessor = DataPreprocessor()
        process_result: ProcessResult = preprocessor.process(data_df, target_column, target_value, min_target_coverage)
        self.column_types: dict[str, str] = process_result.column_types
        self.column_binning: dict[str, bool] = process_result.column_binning
        self.processed_data_df: pd.DataFrame = process_result.data_df

        self.target_column: str = target_column  # Target column is never binned
        self.target_value: Value
        self._init_target_value(target_value, target_column, self.column_types[target_column])
        if not np.any(data_df[target_column].unique() == self.target_value):
            raise Exception('Target value %s can not be found in target column: %s, possible values: %s' %
                            (self.target_value, self.target_column,
                             '[' + ', '.join(data_df[target_column].unique().astype(str)) + ']'))

        data_index: Index = Index(self.processed_data_df, target_column, target_value)
        self.data_index = data_index

    def _init_target_value(self, target_value: Value, target_column: str, target_col_type: str) -> None:
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

    def run(self, mcts_rounds: int = 30000) -> list[ResultPath]:
        tree: MCTSTree | None = MCTSTree(self.data_index, self.column_types, self.min_target_coverage)
        results: list[ResultPath] = tree.run(mcts_rounds)
        del tree
        merger: BinMerger = BinMerger(self.data_index, self.column_types, self.column_binning)
        results = merger.expand(results)
        results = merger.merge(results)
        results = merger.filter(results)
        results = chi2_filter(results)

        # remove duplicated results
        result_map: dict[str, ResultPath] = {}
        for res in results:
            if len(res.items) == 0:
                continue
            if str(res) not in result_map:
                result_map[str(res)] = res
        results = list(result_map.values())
        results = sorted(results, key=lambda r: r.calculate(self.data_index).weight, reverse=True)
        return results
