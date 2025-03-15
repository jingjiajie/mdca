import pandas as pd

data_df: pd.DataFrame = pd.read_csv("pred_2016.csv")

data_df['predict'] = 0
data_df['target'] = data_df['isError']
data_df.drop('isError', axis=1, inplace=True)

# data_df = data_df[data_df['term'] != 6]

data_df.to_csv("pred_2016_processed.csv", index=False)