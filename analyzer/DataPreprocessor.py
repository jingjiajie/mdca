import math
import numpy as np
import pandas as pd

from analyzer.commons import Value

BIN_NUMBER: int = 32  # avoid binning for date
MIN_BIN_STEP: int = 1


class ProcessResult:

    def __init__(self, data_df: pd.DataFrame, column_types: dict[str, str], column_binning: dict[str, bool]):
        self.data_df: pd.DataFrame = data_df
        self.column_types: dict[str, str] = column_types
        self.column_binning: dict[str, bool] = column_binning


class DataPreprocessor:

    def __init__(self):
        pass

    def process(self, data_df: pd.DataFrame, target_column: str, target_value: Value) -> ProcessResult:

        self._drop_single_value_column(data_df)
        data_df.reset_index(drop=True, inplace=True)

        column_types: dict[str, str] = self._infer_and_clean_data_inplace(data_df, data_df.columns)

        column_bin_mode: dict[str, bool] = {}
        for col_name in data_df.columns:
            col_type: str = column_types[col_name]
            if (col_name != target_column and (col_type == 'float' or col_type == 'int')
                    and len(data_df[col_name].unique()) > BIN_NUMBER):
                column_bin_mode[col_name] = True
            else:
                column_bin_mode[col_name] = False

        self._binning_inplace(data_df, column_bin_mode)
        return ProcessResult(data_df, column_types, column_bin_mode)

    def _drop_single_value_column(self, data_df: pd.DataFrame):
        cols_to_drop: list[str] = []
        for col in data_df.columns:
            if len(data_df[col].unique()) == 1:
                cols_to_drop.append(col)
        data_df.drop(cols_to_drop, axis=1, inplace=True)

    @staticmethod
    def _try_convert_float(val: str) -> float | None:
        try:
            return float(val)
        except ValueError:
            return None

    def _infer_and_clean_data_inplace(self, data_df: pd.DataFrame, columns: list[str]) -> dict[str, str]:
        column_types: dict[str, str] = {}
        for col_pos in range(0, len(columns)):
            col_name = columns[col_pos]
            if np.issubdtype(data_df[col_name].dtype, bool):
                column_types[col_name] = 'bool'
            elif np.issubdtype(data_df[col_name].dtype, int):
                column_types[col_name] = 'int'
            elif np.issubdtype(data_df[col_name].dtype, float):
                unique_values: pd.Series = pd.Series(data_df[col_name].unique())
                non_na_unique_values: pd.Series = unique_values[unique_values.notna()]
                if np.all(np.floor(non_na_unique_values) == non_na_unique_values):
                    column_types[col_name] = 'int'
                    if len(non_na_unique_values) == len(unique_values):
                        data_df[col_name] = data_df[col_name].astype(int)
                else:
                    column_types[col_name] = 'float'
            elif data_df[col_name].dtype == object:
                unique_values: pd.Series = pd.Series(data_df[col_name].unique())
                non_na_unique_values: pd.Series = unique_values[unique_values.notna()]
                # Check bool
                is_bool: bool = True
                for val in non_na_unique_values:
                    val_str: str = str(val)
                    val_str = val_str.strip().lower()
                    if val_str not in ['true', 'false']:
                        is_bool = False
                        break
                if is_bool:
                    column_types[col_name] = 'bool'
                    if len(unique_values) == len(non_na_unique_values):
                        data_df[col_name] = data_df[col_name].astype(bool)
                    else:
                        replace_map: dict = {}
                        for val in non_na_unique_values:
                            if type(val) is bool:
                                continue
                            val_bool: bool = str(val).strip().lower() == 'true'
                            replace_map[val] = val_bool
                        replace_map[np.nan] = None
                        data_df.replace({col_name: replace_map}, inplace=True)
                    continue

                # Check int/float
                is_numeric: bool = True
                is_int: bool = True
                sas_missing_values: list[str] = []
                for val in non_na_unique_values:
                    val_str: str = str(val)
                    float_val: float | None = self._try_convert_float(val_str)
                    if float_val is not None:
                        if int(float_val) != float_val:
                            is_int = False
                        continue
                    elif val_str.strip() == '.':
                        sas_missing_values.append(val_str)
                        continue
                    else:
                        is_numeric = False
                        break
                if is_numeric:
                    if len(sas_missing_values) > 0:
                        replace_map: dict[str, float] = {}
                        for item in sas_missing_values:
                            replace_map[item] = np.nan
                        data_df.replace({col_name: replace_map}, inplace=True)
                    if is_int:
                        column_types[col_name] = 'int'
                        if np.all(pd.Series(data_df[col_name].unique()).notna()):
                            data_df[col_name] = data_df[col_name].astype(int)
                        else:
                            data_df[col_name] = data_df[col_name].astype(float)
                    else:
                        column_types[col_name] = 'float'
                        data_df[col_name] = data_df[col_name].astype(float)
                    continue

                # String type
                column_types[col_name] = 'str'
                data_df.replace({col_name: {np.nan: None}}, inplace=True)
        return column_types

    def _binning_inplace(self, data_df: pd.DataFrame, column_bin_mode: dict[str, bool]):
        for col_name in [c for c in column_bin_mode.keys() if column_bin_mode[c]]:
            [q00, q01, q99, q100] = data_df[col_name].quantile(q=[0, 0.01, 0.99, 1]).reset_index(drop=True)
            q00_int: int = math.floor(q00)
            q01_int: int = math.floor(q01)
            q99_int: int = math.ceil(q99)
            q100_int: int = math.ceil(q100)
            if q100 == q100_int:
                q100_int += 1
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
