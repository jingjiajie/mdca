import math

import numpy as np
import pandas as pd

BIN_NUMBER: int = 50
MIN_BIN: int = 1

SORT_UNIQUE_VALUES_THRESHOLD = 20


class DataPreprocessor:

    def __init__(self):
        pass

    def process_inplace(self, data_df: pd.DataFrame, target_column: str | None) -> None:
        data_df.reset_index(drop=True, inplace=True)
        if target_column is not None:
            data_df.drop(target_column, axis=1, inplace=True)

        column_types: list[str] = []
        for col in data_df.columns:
            guess_type: str = self._determine_column_type(data_df, col)
            column_types.append(guess_type)
        column_types: list[str] = column_types

        columns_for_binning: list[str] = []
        for i in range(0, len(column_types)):
            col_type: str = column_types[i]
            col_name: str = data_df.columns[i]
            if col_type == 'float' and len(data_df[col_name].unique()) > BIN_NUMBER:
                columns_for_binning.append(col_name)

        self._convert_df_values_inplace(data_df, data_df.columns, column_types)
        self._binning_inplace(data_df, columns_for_binning)
        self._sort_by_unique_values_inplace(data_df)
        self._stringify_df_values_inplace(data_df)

    def _determine_column_type(self, data_df: pd.DataFrame, column: str, sampling_count: int = 2000) -> str:
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

    def _convert_df_values_inplace(self, data_df: pd.DataFrame, columns: list[str], types: list[str]):
        for i in range(0, len(columns)):
            col = columns[i]
            col_type = types[i]

            def _clean_value(val):
                if col_type == 'float':
                    if type(val) is str and val.strip() in ['', '.']:
                        return float('nan')
                return val

            data_df[col] = data_df[col].map(_clean_value)
            if col_type == 'str':
                data_df[col] = data_df[col].astype(str)
            elif col_type == 'float':
                data_df[col] = data_df[col].astype(float)

    def _stringify_df_values_inplace(self, data_df: pd.DataFrame):
        def _process_value(val) -> str:
            if type(val) is float:
                if np.isnan(val):
                    val = "nan"
                else:
                    val = str(val)
                    if "." in val:
                        val = val.rstrip('0')
                        if val.endswith("."):
                            val = val[:-1]
            else:
                val = str(val)
            return val

        for col in data_df.columns:
            data_df[col] = data_df[col].map(_process_value)
            data_df[col].astype(str)

    def _binning_inplace(self, data_df: pd.DataFrame, bin_cols: list[str]):
        for col_name in bin_cols:
            min_int: int = math.floor(data_df[col_name].min())
            max_int: int = math.ceil(data_df[col_name].max())
            step: float = (max_int - min_int) / BIN_NUMBER
            if step < MIN_BIN:
                step = MIN_BIN
            bins: list[int] = []
            cur_bin: float = min_int
            while cur_bin <= max_int:
                if max_int - cur_bin < MIN_BIN:
                    bins.append(max_int)
                else:
                    bins.append(math.floor(cur_bin))
                cur_bin += step
            data_df[col_name] = pd.cut(data_df[col_name], bins=bins, include_lowest=True, right=False)

    def _sort_by_unique_values_inplace(self, data_df: pd.DataFrame):
        unique_values: list[(str, int)] = []
        for col in data_df.columns:
            unique: int = len(data_df[col].unique())
            unique_values.append((col, unique))
        unique_values.sort(key=lambda item: item[1])
        sort_column_order: list[str] = list(
            map(lambda item: item[0],
                filter(lambda item: item[1] <= SORT_UNIQUE_VALUES_THRESHOLD, unique_values)))
        data_df.sort_values(sort_column_order, inplace=True, ignore_index=True)
