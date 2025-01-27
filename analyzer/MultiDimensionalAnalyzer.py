import time

import pandas as pd

from analyzer.Index import Index
from analyzer.MCTSTree import MCTSTree
from analyzer.ResultPath import ResultPath
from analyzer.DataPreprocessor import DataPreprocessor
from analyzer.chi2_filter import chi2_filter

BIN_NUMBER: int = 50
MIN_BIN: int = 1

SORT_UNIQUE_VALUES_THRESHOLD = 20


class MultiDimensionalAnalyzer:

    def __init__(self, data_df: pd.DataFrame, ignore_columns: list[str] | None = None, affect_threshold_ratio: float = 0.1):
        self._affect_threshold_count: int = int(len(data_df) * affect_threshold_ratio)

        preprocessor: DataPreprocessor = DataPreprocessor()
        preprocessor.process_inplace(data_df, ignore_columns, is_sas_dataset=True)
        self._processed_data_df: pd.DataFrame = data_df

        index: Index = Index(data_df, self._affect_threshold_count)
        self._index = index

    def run(self, mcts_rounds: int = 10000) -> list[ResultPath]:
        tree: MCTSTree = MCTSTree(self._index, self._affect_threshold_count)
        start_time: float = time.time()
        results: list[ResultPath] = tree.run(mcts_rounds)
        print("MCTS cost: %.2f seconds" % (time.time() - start_time))

        start_time = time.time()
        results = chi2_filter(results, self._processed_data_df, self._index)
        print("Chi2 test cost: %.2f seconds" % (time.time() - start_time))

        # remove duplicated results
        result_map: dict[str, ResultPath] = {}
        for res in results:
            if len(res.items) == 0:
                continue
            if str(res) not in result_map:
                result_map[str(res)] = res

        results = list(result_map.values())
        return results
