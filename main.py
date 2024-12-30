import random
import time

import pandas as pd

from binning import binning_inplace
from Index import Index

pd.set_option('expand_frame_repr', False)

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
index: Index = Index(data_df, ["LOAN","MORTDUE","VALUE","REASON","JOB","YOJ","DEROG","DELINQ","CLAGE","NINQ","CLNO","DEBTINC"])

random.seed(time.time())


selected = {}
for i in range(1, 2381):
    val = index.random_select_by_freq('JOB')
    if val not in selected:
        selected[val] = 0
    selected[val]+=1
print(selected)
