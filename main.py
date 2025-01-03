import random
import time
from typing import Iterable

import pandas as pd

from MCTSTree import MCTSTree
from MCTSTreeNode import MCTSTreeNode
from binning import binning_inplace
from Index import Index, IndexLocationList, IndexLocation

pd.set_option('expand_frame_repr', False)

THRESHOLD = 100

data_df: pd.DataFrame = pd.read_csv('data/hmeq/hmeq_train_converted.csv')
# hmeq_train_df = hmeq_train_df.map(lambda item: np.nan if str(item).strip() == '.' else item)
# hmeq_train_df.to_csv('data/hmeq/hmeq_train_converted.csv')
# TODO 提前指定目标列？
data_df.drop("BAD", axis=1, inplace=True)
# TODO 自动确定分桶列
binning_inplace(data_df, ["LOAN","MORTDUE","VALUE","YOJ","DEROG","DELINQ","CLAGE","NINQ","CLNO","DEBTINC"])
# TODO 按distinct value数量决定排序顺序
data_df.sort_values(["REASON","JOB","LOAN","MORTDUE","VALUE","YOJ","DEROG","DELINQ","CLAGE","NINQ","CLNO","DEBTINC"], inplace=True)
# TODO 自动确定索引列
index: Index = Index(data_df, ["LOAN","MORTDUE","VALUE","REASON","JOB","YOJ","DEROG","DELINQ","CLAGE","NINQ","CLNO","DEBTINC"], THRESHOLD)

random.seed(time.time())

# [REASON=DebtCon,NINQ=[1.0, 2.0)]
loc: IndexLocationList = (index._index["REASON"]["DebtCon"]
       .intersect(index._index["NINQ"]["[1.0, 2.0)"])
        .intersect(index._index["JOB"]["Other"])
       )

print(loc.count)

start = time.time()
tree: MCTSTree = MCTSTree(index, THRESHOLD)
result = tree.run(100)
item: MCTSTreeNode
end = time.time()
print("MCTS cost seconds: ", end-start)
for item in result:
    print(item.path(), item.influence_count)


