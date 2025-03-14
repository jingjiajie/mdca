# MDCA：多维数据组合分析 #

## MDCA是什么? ##
MDCA通过多维数据组合分析数据表。支持多维分布、公平性和模型误差分析。

### 多维分布分析 ###
数据的分布偏差可能导致预测模型偏向多数类和过拟合少数类，从而影响模型的准确性。即使每列的不同值的数据分布是均匀的，多列中的值的组合也往往是不均匀的。
多维分布分析可以快速找到偏离基线分布的值组合。  

### 多维公平性分析 ###
数据可能天生就有偏见。例如，性别、种族和国籍值可能会导致模型做出有偏差的预测，简单地删除可能有偏差的列并不总是可行的。
即使每一列都是公平的，多列的组合也可能有偏见。多维公平性分析可以快速找到偏离基线阳性率和较高金额的值组合。
  
现已支持数据集中的公平性检测，但模型公平性（例如Equal Odds, Demographic Parity等）正在开发中。

### 多维模型误差分析 ###
模型对不同的值组合具有不同的预测精度。寻找预测错误率较高的值组合有助于理解模型的误差，从而提高数据质量，提高模型预测精度。
多维模型误差分析可以快速找到预测错误率偏离基线和预测误差较大的值组合。

## 安装 ##
```bash
pip install mdca
```

## 基本用法 ##

### 多维分布分析 ###
```bash
mdca --data='path/to/data.csv' --mode=distribution --min-coverage=0.05  
mdca --data='path/to/data.csv' --mode=distribution --min-coverage=0.05 --target-column=label --target-value=1  
```

### 多维公平性分析 ###
```bash
mdca --data='path/to/data.csv' --mode=fairness --target-column=label --target-value=true --min-coverage=0.05  
```

### 多维模型错误分析 ###
```bash
mdca --data='path/to/data.csv' --mode=error --target-column=label --prediction-column=label_pred --min-error-coverage=0.05  
```
