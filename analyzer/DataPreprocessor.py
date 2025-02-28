import math
import time

import numpy as np
import pandas as pd

from analyzer.commons import Value, ColumnInfo

BIN_NUMBER: int = 32  # avoid binning for date
MIN_BIN_STEP: int = 1


class ProcessResult:

    def __init__(self, data_df: pd.DataFrame, column_info: dict[str, ColumnInfo]):
        self.data_df: pd.DataFrame = data_df
        self.column_info: dict[str, ColumnInfo] = column_info


class DataPreprocessor:

    def __init__(self):
        pass

    def process(self, data_df: pd.DataFrame, target_column: str | None,
                target_value: Value | None, min_coverage: float, search_mode: str) -> ProcessResult:
        print("Preprocessing data...")
        start: float = time.time()
        min_count: int
        if search_mode == 'fairness':
            target_count: int = np.count_nonzero(data_df[target_column] == target_value)
            min_count = int(target_count * min_coverage)
        elif search_mode == 'distribution':
            min_count = int(len(data_df) * min_coverage)
        single_value_columns: list[str] = []
        for col_name in data_df.columns:
            unique_values: np.ndarray = data_df[col_name].unique()
            if len(unique_values) == 1:
                single_value_columns.append(col_name)
        data_df.drop(single_value_columns, axis=1, inplace=True)

        column_types: dict[str, str] = self._infer_and_clean_data_inplace(data_df, data_df.columns)

        column_binning: dict[str, bool] = {}
        for col_name in data_df.columns:
            col_type: str = column_types[col_name]
            if (col_name != target_column and (col_type == 'float' or col_type == 'int')
                    and len(data_df[col_name].unique()) > BIN_NUMBER):
                column_binning[col_name] = True
            else:
                column_binning[col_name] = False

        too_few_value_count_columns: list[str] = []
        for col_name in data_df.columns:
            if column_binning[col_name]:
                continue
            value_counts: pd.Series = data_df[col_name].value_counts()
            if (len(value_counts) == len(data_df) or
                    np.count_nonzero(value_counts < min_count) == len(value_counts)):
                too_few_value_count_columns.append(col_name)
        data_df.drop(too_few_value_count_columns, axis=1, inplace=True)
        print(" - Ignored columns:", '[' + ', '.join(single_value_columns + too_few_value_count_columns) + ']')
        print(" - Inferred column types: %s" %
              ", ".join(map(lambda item: "(%s: %s)" % (item[0], item[1]), column_types.items())))
        print(" - Binning columns: %s" %
              '[' +
              ', '.join(map(lambda item: item[0],
                            filter(lambda item: item[1], column_binning.items())))
              + ']'
              )

        column_q00: dict[str, float] = {}
        column_q01: dict[str, float] = {}
        column_q99: dict[str, float] = {}
        column_q100: dict[str, float] = {}
        for col in data_df.columns:
            if column_types[col] in ['int', 'float']:
                [q00, q01, q99, q100] = data_df[col].quantile(q=[0, 0.01, 0.99, 1]).reset_index(drop=True)
                column_q00[col] = q00
                column_q01[col] = q01
                column_q99[col] = q99
                column_q100[col] = q100

        column_info: dict[str, ColumnInfo] = {}
        for col in data_df.columns:
            info: ColumnInfo
            if column_types[col] in ['int', 'float']:
                info = ColumnInfo(col, column_types[col], column_binning[col], column_q00[col],
                                  column_q01[col], column_q99[col], column_q100[col])
            else:
                info = ColumnInfo(col, column_types[col], column_binning[col], None, None, None, None)
            column_info[col] = info
        self._binning_inplace(data_df, column_info)
        print("Preprocess data cost: %.2f seconds" % (time.time() - start))
        return ProcessResult(data_df, column_info)

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
                non_na_unique_values: list = []
                for val in unique_values:
                    if np.issubdtype(type(val), float) and np.isnan(val):
                        continue
                    elif np.issubdtype(type(val), str) and val.strip().lower() == 'nan':
                        continue
                    else:
                        non_na_unique_values.append(val)

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

    def _binning_inplace(self, data_df: pd.DataFrame, column_info: dict[str, ColumnInfo]):
        for col_name, col_info in column_info.items():
            if not col_info.binning:
                continue
            q00_int: int = math.floor(col_info.q00)
            q01_int: int = math.floor(col_info.q01)
            q99_int: int = math.ceil(col_info.q99)
            q100_int: int = math.ceil(col_info.q100)
            if col_info.q100 == q100_int:
                q100_int += 1
            step: float = (col_info.q99 - col_info.q01) / (BIN_NUMBER - 2)
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
