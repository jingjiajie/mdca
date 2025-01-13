import random
import time

import pandas as pd

from analyzer.MultiDimensionalAnalyzer import MultiDimensionalAnalyzer
from analyzer.ResultPath import ResultPath
from analyzer.Index import IndexLocationList

random.seed(time.time())

# data_df: pd.DataFrame = pd.read_csv('data/hmeq/hmeq_train.csv')
data_df: pd.DataFrame = pd.read_csv('data/flights/flights.csv')

data_df.dropna(inplace=True, subset=['AIR_SYSTEM_DELAY'])

analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, "AIR_SYSTEM_DELAY")
results: list[ResultPath] = analyzer.run()
for r in results:
    loc: IndexLocationList | None = None
    for item in r.items:
        if loc is None:
            loc = analyzer._index.get_locations(item.column, item.value)
        else:
            loc = loc.intersect(analyzer._index.get_locations(item.column, item.value))
    if loc is None:
        print(r, 0)
    else:
        print(r, loc.count)
