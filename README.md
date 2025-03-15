# MDCA: Multi-dimensional Data Combination Analysis

## Languages 多语言：
#### [English Version](README_en.md)  ####
#### [简体中文版本](README.md)  ####

## 什么是MDCA?

MDCA（多维数据组合分析）对数据表中的多个维度的数据组合进行分析。支持多维分布分析、公平性分析和模型误差分析。

### 多维分布分析

数据的分布偏差可能会导致预测模型偏向于多数类，并对少数类产生过拟合，从而影响模型的准确性。即使每一列中不同值的分布是均匀的，多列值的组合往往也会呈现非均匀性。
多维分布分析能够快速发现偏离基线分布的值组合。

### 多维公平性分析

数据本身可能带有偏见。例如，性别、种族和国籍等值可能会导致模型做出有偏见的预测，而且简单地移除可能带有偏见的列并不总是可行的。即使每一列都是公平的，多列的组合也可能产生偏见。
多维公平性分析能够快速发现那些偏离基线正例率以及正例数量较高的值组合。

目前支持对原始数据集进行公平性分析，但模型的公平性（如平等机会、人口统计均等性等）仍在开发中。

### 多维模型误差分析

对于不同值组合，模型预测的准确性会有不同。找到预测错误率较高的值组合有助于了解模型的错误，从而可以提高数据质量，进而提高模型预测的准确性。
多维模型误差分析能够快速发现那些偏离基线，预测错误率以及预测错误数量较高的值组合。

## 安装

```bash
pip install mdca
```

## 典型用法

### 分布分析

```bash
# 推荐用法
mdca --data='path/to/data.csv' --mode=distribution --min-coverage=0.05 --target-column=<name of label column> --target-value=<value of positive label>  

# 不指定目标列和目标值
mdca --data='path/to/data.csv' --mode=distribution --min-coverage=0.05  
```

### 公平性分析

```bash
mdca --data='path/to/data.csv' --mode=fairness --target-column=<name of label column> --target-value=<value of positive label> --min-coverage=0.05  
```

### 模型误差分析

```bash
mdca --data='path/to/data.csv' --mode=error --target-column=<name of label column> --prediction-column=<name of predicted label column> --min-error-coverage=0.05  
```

## 相关概念

对于一个数据表来说，它通常包含多个列，用于描述对象的多种特性。在某些情况下，当这些数据被用于训练分类模型时，会存在一个“实际标签”（actual label）列，该列记录了每个对象的真实类别。此外，在进行模型预测时，还会存在一个“预测标签”（predicted label）列，用于存储模型对每个对象的预测结果。


| columnA | columnB | ... | columnX | actual label<br/>(optional) | predicted label<br/>(optional) |
| ------- | ------- | --- | ------- | --------------------------- | ------------------------------ |
| valueA1 | valueB1 | ... | valueX1 | 1                           | 1                              |
| valueA2 | valueB2 | ... | valueX2 | 0                           | 1                              |
| valueA3 | valueB3 | ... | valueX3 | 0                           | 0                              |
| valueA4 | valueB4 | ... | valueX4 | 1                           | 1                              |
| ...     | ...     | ... | ...     | ...                         | ...                            |

对于这种类型的数据表，MDCA会使用以下概念：

**Target column** (-tc or --target-column)：目标列（通常是实际标签列）。 在 **_distribution_** （分布分析）模式下，实际标签列是可选的，但是在 **_fairness_** （公平性分析）和 **_error_** （误差分析）模式下，它是必需的。

**Target value** (-tv or --target-value)：目标列中正样本的标签值，用于标识属于某一特定类别的样本。对于二元分类问题，常用的正样本标签值包括“1”、“true”，它们代表一个明确的类别；对于多分类问题，可以指定想要分析的任何一个目标类别，例如新闻分类的“体育”或者天气预报中的“雨天”都可以被指定为正样本的标签值。

**Prediction column** (-pc or --prediction-column)：预测标签列。主要应用于 **_error_** （误差分析）模式，在这个模式下，我们需要对比模型的预测结果与实际结果，以评估模型性能。

**Min coverage** (-mc or --min-coverage)：在分析中被考虑的值组合在总数据中所占的最小行数比例。低于这个阈值的数据组合将被忽略。可以使用 _mdca --help_ 查看默认值

**Min target coverage** (-mtc or --min-target-coverage)：目标数据中（目标列中的值等于目标值）被分析的值组合所占行的最小比例。低于这个阈值的数据组合将被忽略。可以使用 _mdca --help_ 查看默认值

**Min error coverage** (-mec or --min-error-coverage)：在误差数据中，被分析的值组合所占行的最小比例（即预测列中的值不等于目标列中的值）。低于这个阈值的数据组合将被忽略。可以使用 _mdca --help_ 查看默认值

## 入门指南

### 进行分布分析

要进行 _分布分析_ ，您需要指定一个数据表路径（目前支持CSV格式）并将分析模式设置为“distribution”（分布）。如果您的数据表包含目标列，则建议使用 **_target-column_** 和 **_target-value_** 两个参数指定目标列和目标值。这样，分析器可以为每个分布提供与目标相关的指标。
快捷指令如下：
```bash
# 推荐用法
mdca --data='path/to/data.csv' --mode=distribution --target-column=<name of label column> --target-value=<value of positive label>

# 不指定目标列和目标值
mdca --data='path/to/data.csv' --mode=distribution

```

**_Min coverage_** 是必需的，但是如果您不指定具体值，则默认使用--help中描述的默认值。  
你也可以手动指定参数，如最小覆盖率、最小目标覆盖率：
```bash
# 手动指定最小覆盖率/最小目标覆盖率
mdca --data='path/to/data.csv' --mode=distribution --min-coverage=0.05  
mdca --data='path/to/data.csv' --mode=distribution --min-target-coverage=0.05  
```

您还可以指定想要分析的某一列或某几列：
```bash
# 如果您希望确保column1、column2、column3是均匀分布的
mdca --data='path/to/data.csv' --mode=distribution --column='column1, column2, column3'  
```

执行完成后，您将得到类似下面的结果：

========== Results of Coverage Increase ============


| Coverage (Baseline, +N%, *X)     | Target Rate(Overall +%N) | Result                                                                                          |
| -------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------- |
| 54.52% ( 8.33%, +46.19%, *6.54 ) | 25.95% ( -5.72%)         | [nationality=Dutch, ind-debateclub=False, ind-entrepeneur_exp=False]                            |
| 62.00% (16.67%, +45.33%, *3.72 ) | 29.35% ( -2.32%)         | [nationality=Dutch, ind-international_exp=False]                                                |
| 41.33% (11.11%, +30.21%, *3.72 ) | 35.63% ( +3.96%)         | [gender=male, nationality=Dutch]                                                                |
| 39.40% (11.11%, +28.29%, *3.55 ) | 20.69% (-10.99%)         | [nationality=Dutch, ind-degree=bachelor]                                                        |
| 30.33% ( 4.17%, +26.16%, *7.28 ) | 26.30% ( -5.38%)         | [ind-debateclub=False, ind-international_exp=False, ind-entrepeneur_exp=False, ind-languages=1] |
| ...                              | ...                      | ...                                                                                             |

在这组结果中，包含以下三列： __Coverage (Baseline, +N%, *X)__ ， **Target Rate(Overall +N%)** ，和 **Result** 。  
**Coverage** 指当前结果中的行数中的行数在总数据中所占的实际比例. （ **Baseline** 指当前结果的预期覆盖率。 __+N%, *X__ 指实际覆盖率是多少，以及比基线覆盖率高出多少倍。）    
**Target Rate** 指在给定的值组合中，正样本的比率。  
**Result** 指给定的值组合。

上面提到的 **Baseline** （基线）覆盖率是通过以下公式计算的：

$$
\vec{C} = (column1, column2, ..., columnN) ∈ Columns(Data Table)
$$

$$
Baseline Coverage(\vec{C}) = \frac{1}{Unique Value Combinations(\vec{C})}
$$

举个例子，性别有两个值： *male*， *female*，国籍中也有两个值： *China*， *America*。 那么列可以是：

$$
\vec{C}=(gender, nationality)
$$

值组合有： {*(male, China)， (male, America)， (female, China)， (female, America)*}。
唯一值组合的长度为4。

$$
Unique Value Combinations(\vec{C}) = 4
$$

那么基线覆盖率按照如下公式进行计算：

$$
Baseline Coverage(\vec{C}) = \frac{1}{4} = 0.25
$$

该算法表明，基线覆盖率是指在所有数据理想均匀分布的情况下，某一值组合的行数所占的比例。

### 进行公平性分析

要进行 _公平性分析_ ，您需要指定一个数据表路径（目前支持CSV格式）并将分析模式设置为“fairness”（公平性）。
同时，需要使用 **_target-column_** 和 **_target-value_** 两个参数指定目标列和目标值，这两个参数是必需的，这样可以分析每个值组合对应的目标率的公平性。
快捷指令如下：
```bash
mdca --data='path/to/data.csv' --mode=fairness --target-column=<name of label column> --target-value=<value of positive label>
```
**_Min coverage_** 是必需的，但您如果不指定具体值，默认使用--help中描述的默认值。  
你也可以手动指定参数，如最小覆盖率、最小目标覆盖率：
```bash
mdca --data='path/to/data.csv' --mode=fairness  --target-column=<name of label column> --target-value=<value of positive label> --min-coverage=0.05  
mdca --data='path/to/data.csv' --mode=fairness  --target-column=<name of label column> --target-value=<value of positive label> --min-target-coverage=0.05  
```

您还可以指定想要分析的某一列或某几列：
```bash
# 如果您希望确保column1、column2和column3的组合的正样本率是公平的
mdca --data='path/to/data.csv' --mode=fairness --column='column1, column2, column3' --target-column=<name of label column> --target-value=<value of positive label>  
```

执行完成后，您将得到类似下面的结果：

========== Results of Target Rate Increase ============

| Coverage(Count), | Target Rate(Overall+N%), | Result                           |
|------------------|--------------------------|----------------------------------|
| 13.18% (   527), | 41.75% (+10.07%),        | [gender=male, sport=Rugby]       |
| 5.33% (   213),  | 44.13% (+12.46%),        | [gender=male, age=29]            |
| 7.22% (   289),  | 40.14% ( +8.46%),        | [age=30]                         |
| 41.33% (  1653), | 35.63% ( +3.96%),        | [gender=male, nationality=Dutch] |
| 15.72% (   629), | 36.09% ( +4.41%),        | [gender=male, sport=Football]    |
| 5.92% (   237),  | 37.55% ( +5.88%),        | [gender=male, age=24]            |
| ...              | ...                      | ...                              |

在这组结果中，包含以下三列： **Coverage (Count)**， **Target Rate(Overall +N%)**，和 **Result**。  
**Coverage** 指在当前结果中，各组合所占的行数比例与总数据中各组合所占的比例应该是一致的。（**Count** 指实际的行数比例。）  
**Target Rate** 指在给定的值组合数据中，正样本所占的比例。（ **(Overall +N%)** 指目标率相对于总数据表中整体目标率的偏离程度。）  
**Result** 指给定的值组合数据。 


### 进行模型误差分析

要进行 _模型误差分析_ ，您需要指定一个数据表路径（目前支持CSV格式）并将分析模式设置为“error”（误差）。
同时，需要使用 **_target-column_** 和 **_target-value_** 两个参数指定目标列和目标值，这两个参数是必需的，这样可以分析每个值组合的错误率了。
快捷指令如下：
```bash
mdca --data='path/to/data.csv' --mode=error --target-column=<name of label column> --prediction-column=<name of predicted label column> 
```
**_Min error coverage_** 是必需的，但您如果不指定具体值，默认使用--help中描述的默认值。  
你也可以手动指定参数，如最小覆盖率、最小目标覆盖率：
```bash
mdca --data='path/to/data.csv' --mode=error  --target-column=<name of label column> --prediction-column=<name of predicted label column>  --min-coverage=0.05  
mdca --data='path/to/data.csv' --mode=error  --target-column=<name of label column> --prediction-column=<name of predicted label column>  --min-error-coverage=0.05  
```

您还可以指定想要分析的某一列或某几列：
```bash
# 如果您希望对column1、column2和column3的组合进行误差分析
mdca --data='path/to/data.csv' --mode=error --column='column1, column2, column3' --target-column=<name of label column> --prediction-column=<name of predicted label column>
```

执行完成后，您将得到类似下面的结果：

========== Results of Error Rate Increase ============

| Error Coverage(Count) | Error Rate(Overall+N%) | Result                                           |
|-----------------------|------------------------|--------------------------------------------------|
| 51.69% ( 20713)       | 35.97% (+12.92%)       | [subGrade_trans=[14, 30)]                        |
| 11.46% (  4591)       | 40.35% (+17.31%)       | [term=5, verificationStatus=2]                   |
| 12.22% (  4897)       | 36.36% (+13.32%)       | [term=5, verificationStatus=1]                   |
| 21.04% (  8430)       | 32.77% ( +9.73%)       | [verificationStatus=2, ficoRangeHigh=[664, 687)] |
| 5.90% (  2364)        | 37.13% (+14.08%)       | [term=5, n14=3]                                  |
| 53.32% ( 21365)       | 28.40% ( +5.36%)       | [ficoRangeHigh=[664, 687)]                       |
| ...                   | ...                    | ...                                              |

在这组结果中，包含以下三列： **Error Coverage (Count)**, **Error Rate(Overall +N%)**，和 **Result**。  
**Error Coverage** 指当前结果在预测错误数据中所占的实际行数比例。（**Count** 指实际行数。）  
**Error Rate** 指在给定值组合的数据中预测错误的比率。（**(Overall +N%)** 指错误率比总数据表中的整体错误率高出了多少。）
**Result** 指给定的值组合数据。 

## 问题报告 & 帮助

欢迎报告任何bug或功能需求至：[https://github.com/jingjiajie/mdca/issues](https://github.com/jingjiajie/mdca/issues)  
管理员会尽快回复的。

如果您需要任何帮助， 可以发送邮件至作者的邮箱： **932166095@qq.com** 或者添加微信： **18515221942**  
作者会尽快给予快速帮助。 