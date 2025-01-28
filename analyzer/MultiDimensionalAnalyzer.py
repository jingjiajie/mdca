import time

import pandas as pd

from analyzer.Index import Index
from analyzer.MCTSTree import MCTSTree
from analyzer.ResultPath import ResultPath, ResultItem
from analyzer.DataPreprocessor import DataPreprocessor
from analyzer.chi2_filter import chi2_filter

BIN_NUMBER: int = 50
MIN_BIN: int = 1

SORT_UNIQUE_VALUES_THRESHOLD = 20


class MultiDimensionalAnalyzer:

    def __init__(self, data_df: pd.DataFrame, target_column: str, target_value: str,
                 affect_threshold_ratio: float = 0.05):
        self.target_column: str = target_column
        self.target_value: str = target_value
        self._affect_threshold_count: int = int(len(data_df) * affect_threshold_ratio)
        # TODO 先筛选后处理
        preprocessor: DataPreprocessor = DataPreprocessor()
        preprocessor.process_inplace(data_df, is_sas_dataset=True)
        self._processed_full_data_df: pd.DataFrame = data_df

        target_df: pd.DataFrame = data_df[data_df[target_column] == target_value].copy()
        target_df.drop(target_column, axis=1, inplace=True)
        target_df.reset_index(drop=True, inplace=True)
        self._processed_target_df: pd.DataFrame = target_df

        target_df_index: Index = Index(self._processed_target_df, self._affect_threshold_count)
        full_index: Index = Index(self._processed_full_data_df, self._affect_threshold_count)
        self._target_index = target_df_index
        self._full_index = full_index

    def run(self, mcts_rounds: int = 10000) -> list[ResultPath]:
        tree: MCTSTree = MCTSTree(self._target_index, self._affect_threshold_count)
        start_time: float = time.time()
        results: list[ResultPath] = tree.run(mcts_rounds)
        for res_path in results:
            res_path.items.insert(0, ResultItem(self.target_column, self.target_value))
        print("MCTS cost: %.2f seconds" % (time.time() - start_time))

        start_time = time.time()
        results = chi2_filter(results, self.target_column, self._full_index)
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
