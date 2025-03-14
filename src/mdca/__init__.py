import argparse
import random
import time

import pandas as pd
from mdca.analyzer.MultiDimensionalAnalyzer import MultiDimensionalAnalyzer
from mdca.analyzer.ResultPath import CalculatedResult

DEFAULT_MIN_COVERAGE: float = 0.05
DEFAULT_MIN_ERROR_COVERAGE: float = 0.05
DEFAULT_MAX_RESULTS: int = 20
DEFAULT_SEARCH_ROUNDS: int = 100000


def _print_help(parser: argparse.ArgumentParser):
    print("""====================================================
 MDCA: Multi-dimensional Data Combination Analysis 
====================================================
MDCA analysis data table through multi-dimensional data combinations.
Multi-dimensional distribution, fairness, and model error analysis are supported.

* Multi-dimensional Distribution Analysis *
The distribution deviation of data may cause the prediction model to be biased
towards majority classes and overfit minority classes, which affects the accuracy
of the model. Even if the data distribution of different values for each column is
uniform, combinations of values in multiple columns tend to be non-uniform.
Multi-dimensional distribution analysis can quickly find the value combinations
with deviated-from-baseline distributions.

* Multi-dimensional Fairness Analysis *
Data can be inherently biased. For example, gender, race, and nationality values
may cause the model to make biased predictions, and it is not always feasible to
simply remove columns that may be biased. Even if every column is fair, combination
of multiple columns can be biased. Multi-dimensional fairness analysis can quickly
find the value combinations with deviated-from-baseline positive rates as well as
higher amounts.
Fairness detection in raw data sets is now supported.
Model fairness (eg. Equal Odds, Demographic Parity, etc.) is under development.

* Multi-dimensional Model Error Analysis *
Model has different prediction accuracy for different value combinations. Finding the
value combinations with higher prediction error rate is helpful to understand the error
of model, so as to improve the data quality and improve model prediction accuracy.
Multi-dimensional model error analysis can quickly find the value combinations with 
deviated-from-baseline prediction error rates as well as higher amounts in prediction
error.
""")

    parser.print_help()

    print("""
Typical usages:

Distribution Analysis:
mdca --data='path/to/data.csv' --mode=distribution --min-coverage=0.05
mdca --data='path/to/data.csv' --mode=distribution --min-coverage=0.05 --target-column=label --target-value=1

Fairness Analysis:
mdca --data='path/to/data.csv' --mode=fairness --target-column=label --target-value=true --min-coverage=0.05

Model Error Analysis:
mdca --data='path/to/data.csv' --mode=error --target-column=label --prediction-column=label_pred --min-error-coverage=0.05
""")


def main():
    start: float = time.time()
    random.seed(time.time())

    parser = argparse.ArgumentParser(prog='mdca', add_help=True)
    parser.add_argument("-d", "--data", dest='data', type=str,
                        help="Path to data table file. Example: path/to/data.csv")
    parser.add_argument('-m', "--mode", dest='mode', type=str,
                        help='Analysis mode. Must be distribution, fairness or error')
    parser.add_argument('-c', "--columns", dest='columns', type=str,
                        help='Optional. Columns to analysis in the data table. Example: "col1,col2,...". '
                             'Omit this argument means all columns')
    parser.add_argument('-tc', "--target-column", dest='target_column', type=str,
                        help='The column of data label. Mandatory in fairness, error mode. '
                             'Optional in distribution mode.')
    parser.add_argument('-tv', "--target-value", dest='target_value', type=str,
                        help='The data label value of positive sample (usually True or 1 in binary classification). '
                             'Mandatory in fairness mode. Omit in error mode. Optional in distribution mode.')
    parser.add_argument('-pc', "--prediction-column", dest='prediction_column', type=str,
                        help='The column of model predicted label. Mandatory in error mode. '
                             'Omit in distribution, fairness mode.')
    parser.add_argument('-mc', "--min-coverage", dest='min_coverage', type=float,
                        help='Minimum proportion of rows of analyzed value combinations in the total data. '
                             'Data combinations lower than this threshold will be ignored. '
                             'Default: %.2f in distribution, fairness mode, none in error mode.' % DEFAULT_MIN_COVERAGE)
    parser.add_argument('-mtc', "--min-target-coverage", dest='min_target_coverage', type=float,
                        help='Minimum proportion of rows of analyzed value combinations in the target data. '
                             '(value in target-column == target-value). '
                             'Data combinations lower than this threshold will be ignored. '
                             'Default: none.')
    parser.add_argument('-mec', "--min-error-coverage", dest='min_error_coverage', type=float,
                        help='Minimum proportion of rows of analyzed value combinations in the error data '
                             '(value in prediction-column != value in target-column). '
                             'Data combinations lower than this threshold will be ignored.'
                             'Default: %.2f in error mode.' % DEFAULT_MIN_ERROR_COVERAGE)
    parser.add_argument('-mr', "--max-results", dest='max_results', type=int,
                        help='Maximum number of output results. Default: %d' % DEFAULT_MAX_RESULTS)
    parser.add_argument('-sr', "--search-rounds", dest='search_rounds', type=int,
                        help='Maximum rounds of heuristic search. Default: %d' % DEFAULT_SEARCH_ROUNDS)

    args = parser.parse_args()

    # Display help when no arguments specified
    arg_dict: dict = args.__dict__
    is_all_empty: bool = True
    for value in arg_dict.values():
        if value is not None:
            is_all_empty = False
    if is_all_empty:
        _print_help(parser)
        exit(0)

    # Fill default values
    mode: str = args.mode
    if mode == 'distribution':
        if args.min_coverage is None:
            args.min_coverage = DEFAULT_MIN_COVERAGE
    elif mode == 'fairness':
        if args.min_coverage is None:
            args.min_coverage = DEFAULT_MIN_COVERAGE
    elif mode == 'error':
        if args.min_error_coverage is None:
            args.min_error_coverage = DEFAULT_MIN_ERROR_COVERAGE
    if args.max_results is None:
        args.max_results = DEFAULT_MAX_RESULTS
    if args.search_rounds is None:
        args.search_rounds = DEFAULT_SEARCH_ROUNDS

    # Clean argument values
    for key in arg_dict.keys():
        val = arg_dict[key]
        if isinstance(val, str):
            if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                val = val[1:-1]
                args.__setattr__(key, val)

    # Validity check
    if args.data is None:
        print('-d (or --data) must be specified!')
        exit(1)
    elif args.mode is None:
        print('-m (or --mode) must be specified!')
        exit(1)
    data_path: str = args.data
    if not data_path.endswith('.csv'):
        print('Currently only .csv data is supported! (actual: %s)' % data_path)
        exit(1)
    if mode == 'distribution':
        if args.prediction_column is not None:
            print('--prediction-column (or -pc) should not be specified for search mode: distribution')
            exit(1)
        elif args.target_column is not None and args.target_value is None:
            print('--target-value (or -tv) must be specified when --target-column (or -tc) is specified(%s) '
                  'for search mode: distribution' % args.target_column)
            exit(1)
        elif args.min_error_coverage is not None:
            print('--min-error-coverage (or -mec) should not be specified for search mode: distribution')
            exit(1)
        elif args.min_coverage is None and args.min_target_coverage is None:
            print('At least one of --min-coverage (or -mc) or --min-target-coverage (or -mtc) must be specified'
                  ' for mode: distribution')
            exit(1)
    elif mode == 'fairness':
        if args.target_column is None:
            print('--target-column (or -tc) must be specified for search mode: fairness')
            exit(1)
        elif args.prediction_column is not None:
            print('--prediction-column (or -pc) is not yet supported in current version for mode: fairness')
            exit(1)
        elif args.target_value is None:
            print('--target-value (or -tv) must be specified for mode: fairness')
            exit(1)
        elif args.min_error_coverage is not None:
            print('--min-error-coverage (or -mec) should not be specified for search mode: fairness')
            exit(1)
        elif args.min_coverage is None and args.min_target_coverage is None:
            print('At least one of --min-coverage (or -mc) or --min-target-coverage (or -mtc) must be specified'
                  ' for mode: fairness')
            exit(1)
    elif mode == 'error':
        if args.target_column is None:
            print('--target-column (or -tc) must be specified for search mode: error')
            exit(1)
        elif args.prediction_column is None:
            print('--prediction-column (or -pc) must be specified for mode: error')
            exit(1)
        elif args.target_value is not None:
            print('--target-value (or -tv) should not be specified for mode: error')
            exit(1)
        elif args.min_target_coverage is not None:
            print('--min-target-coverage (or -mtc) should not be specified for search mode: error')
            exit(1)
        elif args.min_coverage is None and args.min_error_coverage is None:
            print('At least one of --min-coverage (or -mc) or --min-error-coverage (or -mec) must be specified'
                  ' for mode: error')
            exit(1)
    else:
        print('--mode (or -m) must be fairness, distribution or error, actual: %s' % mode)

    columns_str: str = args.columns
    columns: list[str] | None = None
    if columns_str is not None and len(columns_str) > 0:
        columns = columns_str.replace(' ', '').split(',')

    print('Loading %s...' % data_path)
    load_data_start: float = time.time()
    # data_df: pd.DataFrame = pl.read_csv(data_path, encoding="utf8-lossy").to_pandas()
    data_df: pd.DataFrame = pd.read_csv(data_path, low_memory=False)
    print('Load data cost: %.2f seconds' % (time.time() - load_data_start))

    analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df,
                                                                  search_mode=mode,
                                                                  columns=columns,
                                                                  target_column=args.target_column,
                                                                  target_value=args.target_value,
                                                                  prediction_column=args.prediction_column,
                                                                  min_coverage=args.min_coverage,
                                                                  min_target_coverage=args.min_target_coverage,
                                                                  min_error_coverage=args.min_error_coverage)

    results: list[CalculatedResult] = analyzer.run(mcts_rounds=args.search_rounds, max_results=args.max_results)
    print("\nTotal time cost: %.2f seconds" % (time.time() - start))
    analyzer.print_results(results)