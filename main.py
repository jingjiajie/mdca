import random
import time

import pandas as pd

from MCTSTree import MCTSTree
from MCTSTreeNode import MCTSTreeNode
from binning import binning_inplace
from Index import Index, IndexLocationList, IndexLocation

pd.set_option('expand_frame_repr', False)

AFFECT_THRESHOLD = 600
MCTS_ROUNDS = 10000

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
index: Index = Index(data_df, ["LOAN","MORTDUE","VALUE","REASON","JOB","YOJ","DEROG","DELINQ","CLAGE","NINQ","CLNO","DEBTINC"], AFFECT_THRESHOLD)

random.seed(time.time())

# [DELINQ=[0.0, 1.0),NINQ=[0.0, 1.0),DEBTINC=nan]
# loc: IndexLocationList = (index._index["DELINQ"]["[0.0, 1.0)"]
#        .intersect(index._index["NINQ"]["[0.0, 1.0)"])
#     .intersect(index._index["DEBTINC"]['nan'])
#       .intersect(index._index["JOB"]["Mgr"])
#        )

# print(loc.count)

start = time.time()
tree: MCTSTree = MCTSTree(index, AFFECT_THRESHOLD)
result = tree.run(MCTS_ROUNDS)
item: MCTSTreeNode
end = time.time()
print("MCTS cost seconds: ", end-start)
for item in result:
    print(item.path(), item.influence_count)


