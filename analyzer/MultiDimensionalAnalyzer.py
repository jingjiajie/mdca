import time

import numpy as np
import pandas as pd

from analyzer.Index import Index
from analyzer.MCTSTree import MCTSTree
from analyzer.ResultPath import ResultPath, ResultItem
from analyzer.DataPreprocessor import DataPreprocessor, ProcessResult
from analyzer.chi2_filter import chi2_filter
from analyzer.types import Value

BIN_NUMBER: int = 50
MIN_BIN: int = 1

SORT_UNIQUE_VALUES_THRESHOLD = 20


class MultiDimensionalAnalyzer:

    def __init__(self, data_df: pd.DataFrame, target_column: str, target_value: Value,
                 min_error_coverage: float = 0.05, is_sas_dataset: bool = False):

        if isinstance(target_value, float) and np.isnan(target_value):
            target_value = np.nan

        self.target_column: str = target_column
        self.target_value: Value = target_value
        self.min_error_coverage: float = min_error_coverage

        preprocessor: DataPreprocessor = DataPreprocessor()
        process_result: ProcessResult = preprocessor.process(data_df, is_sas_dataset=is_sas_dataset)
        self._processed_data_df: pd.DataFrame = process_result.data_df

        data_index: Index = Index(self._processed_data_df, process_result.column_types, target_column,
                                  target_value, self.min_error_coverage)
        self._data_index = data_index

    def run(self, mcts_rounds: int = 10000) -> list[ResultPath]:
        tree: MCTSTree = MCTSTree(self._data_index, self.min_error_coverage)
        start_time: float = time.time()
        results: list[ResultPath] = tree.run(mcts_rounds)
        print("MCTS cost: %.2f seconds" % (time.time() - start_time))

        # start_time = time.time()
        # results = chi2_filter(results, self.target_column, self.target_value, self._full_index)
        # print("Chi2 test cost: %.2f seconds" % (time.time() - start_time))
        #
        # # remove duplicated results
        # result_map: dict[str, ResultPath] = {}
        # for res in results:
        #     if len(res.items) == 0:
        #         continue
        #     if str(res) not in result_map:
        #         result_map[str(res)] = res
        #
        # results = list(result_map.values())
        return results
