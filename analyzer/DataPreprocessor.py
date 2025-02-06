import math
import numpy as np
import pandas as pd

BIN_NUMBER: int = 20
MIN_BIN_STEP: int = 1

SORT_UNIQUE_VALUES_THRESHOLD = 20

class ProcessResult:

    def __init__(self, data_df: pd.DataFrame, column_types: dict[str, str], column_binning: dict[str, bool]):
        self.data_df: pd.DataFrame = data_df
        self.column_types: dict[str, str] = column_types
        self.column_binning: dict[str, bool] = column_binning


class DataPreprocessor:

    def __init__(self):
        pass

    def process(self, data_df: pd.DataFrame, is_sas_dataset: bool = False) -> ProcessResult:

        self._drop_single_value_column(data_df)
        data_df.reset_index(drop=True, inplace=True)

        column_types: dict[str] = {}
        for col in data_df.columns:
            guess_type: str = self._determine_column_type(data_df, col)
            column_types[col] = guess_type

        column_bin_mode: dict[str, bool] = {}
        for col_name in data_df.columns:
            col_type: str = column_types[col_name]
            if (col_type == 'float' or col_type == 'int') and len(data_df[col_name].unique()) > BIN_NUMBER:
                column_bin_mode[col_name] = True
            else:
                column_bin_mode[col_name] = False

        if is_sas_dataset:
            self._process_sas_missing_values_inplace(data_df, data_df.columns, column_types)

        for col_pos in range(0, len(data_df.columns)):
            col_name: str = data_df.columns[col_pos]
            col_type: str = column_types[col_name]
            if col_type == 'float':
                data_df[col_name] = data_df[col_name].astype(float)
            elif col_type == 'int':
                data_df[col_name] = data_df[col_name].astype(int)

        self._binning_inplace(data_df, column_bin_mode)
        return ProcessResult(data_df, column_types, column_bin_mode)

    def _drop_single_value_column(self, data_df: pd.DataFrame):
        cols_to_drop: list[str] = []
        for col in data_df.columns:
            if len(data_df[col].unique()) == 1:
                cols_to_drop.append(col)
        data_df.drop(cols_to_drop, axis=1, inplace=True)

    def _determine_column_type(self, data_df: pd.DataFrame, column: str, sampling_count: int = 5000) -> str:
        if 'int' in str(data_df[column].dtype):
            return 'int'
        if len(data_df) < sampling_count:
            sampling_count = len(data_df)
        sample_rows = np.random.randint(0, len(data_df), sampling_count)
        series: pd.Series = data_df[column]
        guess_type: str | None = None  # float, str

        def str_is_float(s: str):
            try:
                float(s)
                return True
            except ValueError:
                return False

        for row in sample_rows:
            val = series[row]
            if type(val) is str:
                val: str
                if str_is_float(val) or val.strip() in ['.', '']:
                    if guess_type is None:
                        guess_type = 'float'
                else:
                    guess_type = 'str'
                    break
            elif np.isreal(val):
                guess_type = 'float'
        return guess_type

    def _process_sas_missing_values_inplace(self, data_df: pd.DataFrame, columns: list[str], types: dict[str, str]):
        for col_pos in range(0, len(columns)):
            col_name = columns[col_pos]
            col_type = types[col_name]

            if col_type == 'float' or col_type == 'bin':
                def _clean_value(val):
                    if type(val) is str or type(val) is np.str_:
                        for c in val:
                            if c != ' ' and c != '.':
                                return val
                        return float('nan')
                    else:
                        return val
                data_df[col_name] = data_df[col_name].map(_clean_value)
                data_df[col_name] = data_df[col_name].astype(float)

    def _binning_inplace(self, data_df: pd.DataFrame, column_bin_mode: dict[str, bool]):
        for col_name in [c for c in column_bin_mode.keys() if column_bin_mode[c]]:
            [q00, q01, q99, q100] = data_df[col_name].quantile(q=[0, 0.01, 0.99, 1]).reset_index(drop=True)
            q00_int: int = math.floor(q00)
            q01_int: int = math.floor(q01)
            q99_int: int = math.ceil(q99)
            q100_int: int = math.ceil(q100)
            step: float = (q99 - q01) / (BIN_NUMBER - 2)
            if step < MIN_BIN_STEP:
                step = MIN_BIN_STEP
            bins: list[int] = []
            if q00_int != q01_int:
                bins.append(q00_int)
            cur_bin: int = q01_int
            while cur_bin <= q99_int:
                if q99_int - cur_bin < MIN_BIN_STEP:
                    bins.append(q99_int)
                else:
                    bins.append(math.floor(cur_bin))
                cur_bin += step
            if q100_int != q99_int:
                bins.append(q100_int)
            data_df[col_name] = pd.cut(data_df[col_name], bins=bins, include_lowest=True, right=False)
