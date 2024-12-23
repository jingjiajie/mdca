import time

import pandas as pd
import numpy as np
import shap
from mealy import ErrorAnalyzer, ErrorVisualizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)

start = time.time()
df = pd.read_csv("./data/accepted_2007_to_2018Q4.csv")

end = time.time()
print("loading data cost: %f seconds" % (end - start))

# for i in df.select_dtypes('object').columns:
#     print(i,df[i].unique())
#     print(i,df[i].isnull().sum())
Columnstokeep = ['annual_inc','application_type','bc_util','chargeoff_within_12_mths','delinq_2yrs','dti','emp_length','fico_range_high','fico_range_low','grade','home_ownership','installment','int_rate','loan_amnt','loan_status','mort_acc','mths_since_recent_inq','num_op_rev_tl','purpose','revol_util','term','hardship_flag']
df = df[Columnstokeep]

df['term'].replace(' month?', '', regex=True, inplace=True)
df['term'].replace('s','',regex=True, inplace=True)
df['emp_length'].replace(' years', '', regex=True, inplace=True)
df['emp_length'].replace('10+', '11', inplace=True)
df['emp_length'].replace('1 year','1', inplace=True)
df['emp_length'].replace('< 1 year','0', inplace=True)
df['grade'].replace('G','1', inplace=True)
df['grade'].replace('F','2', inplace=True)
df['grade'].replace('E','3', inplace=True)
df['grade'].replace('D','4', inplace=True)
df['grade'].replace('C','5', inplace=True)
df['grade'].replace('B','6', inplace=True)
df['grade'].replace('A','7', inplace=True)
df.loc[:, 'hardship_flag'].replace({'Y': '1', 'N': '0'}, inplace=True)
df['home_ownership'].replace('ANY',np.nan, inplace=True)
df['home_ownership'].replace('NONE',np.nan, inplace=True)
df['home_ownership'].replace('OTHER',np.nan, inplace=True)
df['home_ownership'].replace({'RENT':'1','MORTGAGE':'2','OWN':'3'}, inplace=True)
df['loan_status'].replace({'Fully Paid':'1','Does not meet the credit policy. Status:Fully Paid':'1','Current':'1','Charged Off':'0','Does not meet the credit policy. Status:Charged Off':'0','In Grace Period':np.nan,'Late (31-120 days)':np.nan,'Late (16-30 days)':np.nan,'Default':'0'}, inplace=True)

df.dropna(subset=['term'],inplace=True)
df.dropna(subset=['loan_status'],inplace=True)

df['loan_status'] = df['loan_status'].astype(float)
df['term'] = df['term'].astype(int)
df['home_ownership'] = df['home_ownership'].astype(float)
df['grade'] = df['grade'].astype(float)
df['emp_length'] = df['emp_length'].astype(float)
df.loc[:, 'hardship_flag'] = df.loc[:, 'hardship_flag'].astype(float)

df.dropna(subset=['mths_since_recent_inq','bc_util','num_op_rev_tl','mort_acc','revol_util','dti','chargeoff_within_12_mths','delinq_2yrs','home_ownership'],inplace=True)

Meanemplength = df['emp_length'].mean()
df['emp_length'].fillna(Meanemplength,inplace=True)
df.isnull().sum()

df['Fico Score']=(df['fico_range_high']+df['fico_range_low'])/2
df=df.drop(['fico_range_high','fico_range_low'],axis=1)

df=df.drop(['revol_util'],axis=1)

columns_to_encode = ['application_type', 'purpose']
df_encoded = pd.get_dummies(df, columns=columns_to_encode)
df_encoded=df_encoded.replace({True:1,False:0})

# copied_df = df_encoded
# for i in range(1, 10):
#      copied_df = copied_df.append(df_encoded)
# df_encoded = copied_df

y = df_encoded['loan_status']
x = df_encoded.drop(['loan_status'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
for i in X_train, X_test, y_train, y_test:
    print(i.shape)

# df.hist(bins = 10, figsize = (22,15))

print(df.head(100))

start = time.time()
print("Start training...")
clf = DecisionTreeClassifier(random_state=42)
# Fit the classifier on the PCA-transformed training data
clf.fit(X_train, y_train)
end = time.time()
print("Training cost: %f seconds" % (end-start))

# 创建一个SHAP解释器
explainer = shap.TreeExplainer(clf, X_train)

# 计算SHAP值
shap_values = explainer.shap_values(X_test)

plot = shap.summary_plot(shap_values, X_test)
print(plot)

# # fit an Error Tree on the model performances
# error_analyzer = ErrorAnalyzer(clf, feature_names=clf.feature_names_in_)
# print("Start fitting error_analyzer")
# error_analyzer.fit(X_test, y_test)
# print("Fitting error_analyzer done")
#
# print("Start evaluating error_analyzer")
# # print metrics regarding the Error Tree
# print(error_analyzer.evaluate(X_test, y_test))
#
# # plot the Error Tree
# error_visualizer = ErrorVisualizer(error_analyzer)
# error_visualizer.plot_error_tree()
#
# # return the details on the decision tree "error leaves" (leaves that contain a majority of errors)
# error_analyzer.get_error_leaf_summary(leaf_selector=None, add_path_to_leaves=True)
#
# # plot the feature distributions of samples in the "error leaves"
# # features are ranked by their correlation to error
# error_visualizer.plot_feature_distributions_on_leaves(leaf_selector=None, top_k_features=3)
