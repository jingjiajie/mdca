# MDCA: Multi-dimensional Data Combination Analysis.

## What's MDCA?

MDCA analyzes multi-dimensional data combinations in data table.
Multi-dimensional distribution, fairness, and model error analysis are supported.

### Multi-dimensional Distribution Analysis

The distribution deviation of data may cause the prediction model to be biased towards majority classes and overfit minority classes, which affects the accuracy of the model.
Even if the data distribution of different values for each column is uniform, combinations of values in multiple columns tend to be non-uniform.
Multi-dimensional distribution analysis can quickly find the value combinations with deviated-from-baseline distributions.

### Multi-dimensional Fairness Analysis

Data can be inherently biased. For example, gender, race, and nationality values may cause the model to make biased predictions,
and it is not always feasible to simply remove columns that may be biased.
Even if every column is fair, combination of multiple columns can be biased.
Multi-dimensional fairness analysis can quickly find the value combinations with deviated-from-baseline positive rates as well as higher amounts.

Fairness detection in raw data sets is now supported, but Model fairness (eg. Equal Odds, Demographic Parity, etc.) is under development.

### Multi-dimensional Model Error Analysis

Model has different prediction accuracy for different value combinations.
Finding the value combinations with higher prediction error rate is helpful to understand the error of model, so as to improve the data quality and improve model prediction accuracy.
Multi-dimensional model error analysis can quickly find the value combinations with deviated-from-baseline prediction error rates as well as higher amounts in prediction error.

## Installing

```bash
pip install mdca
```

## Typical usages

### Distribution Analysis

```bash
# recommended
mdca --data='path/to/data.csv' --mode=distribution --min-coverage=0.05 --target-column=label --target-value=1  

# for data tables doesn't have a label column
mdca --data='path/to/data.csv' --mode=distribution --min-coverage=0.05  
```

### Fairness Analysis

```bash
mdca --data='path/to/data.csv' --mode=fairness --target-column=label --target-value=true --min-coverage=0.05  
```

### Model Error Analysis

```bash
mdca --data='path/to/data.csv' --mode=error --target-column=label --prediction-column=label_pred --min-error-coverage=0.05  
```

## Concepts

For a data table, there are multiple columns to describe multiple characteristics of objects.
Optionally there is also an _actual label_ column if the data is used to train classification models.
As well, for model prediction, there also can be a _predicted label_ to store the prediction results of a model.


| columnA | columnB | ... | columnX | actual label<br/>(optional) | predicted label<br/>(optional) |
| ------- | ------- | --- | ------- | --------------------------- | ------------------------------ |
| valueA1 | valueB1 | ... | valueX1 | 1                           | 1                              |
| valueA2 | valueB2 | ... | valueX2 | 0                           | 1                              |
| valueA3 | valueB3 | ... | valueX3 | 0                           | 0                              |
| valueA4 | valueB4 | ... | valueX4 | 1                           | 1                              |
| ...     | ...     | ... | ...     | ...                         | ...                            |

With this kind of data table, MDCA uses the following concepts:

**Target column** (-tc or --target-column): The name of the actual label column. It's optional in **_distribution_** mode,
but mandatory in **_fairness_** and **_error_** mode.

**Target value** (-tv or --target-value): The label value of positive sample in the target column.
For example, _"1", "true"_ is often used for binary-classification, and for multi-classification,
you can specify it as a target category you want to analysis, like "sport" for a news classification,
or "rain" for a weather prediction.

**Prediction column** (-pc or --prediction-column): The name of predicted label column. It's only available in **_error_** mode now.

**Min coverage** (-mc or --min-coverage): Minimum proportion of rows of analyzed value combinations in the total data.
Data combinations lower than this threshold will be ignored. Default value can be viewed using _mdca --help_

**Min target coverage** (-mtc or --min-target-coverage): Minimum proportion of rows of analyzed value combinations in the target data (value in target-column == target-value).
Data combinations lower than this threshold will be ignored. Default value can be viewed using _mdca --help_

**Min error coverage** (-mec or --min-error-coverage): Minimum proportion of rows of analyzed value combinations in the error data (value in prediction-column != value in target-column). Data combinations lower than this threshold will be ignored. Default value can be viewed using _mdca --help_

## Getting Started

### Performing Distribution Analysis

To perform _distribution analysis_, you need to specify a data table path (CSV is supported so far) and an analysis mode as "distribution".
Meanwhile, **_Target column_** and **_Target value_** are recommended to specify if your data table has a target column.
In this way, analyzer can give target related indicators with each distribution.
The simplest command is:
```bash
# recommended
mdca --data='path/to/data.csv' --mode=distribution --target-column=label --target-value=1

# for data tables doesn't have a label column
mdca --data='path/to/data.csv' --mode=distribution

```

**_Min coverage_** is mandatory, but without specifying a value, it will use a default value described in --help.
You can still manually specify arguments like min coverage, min target coverage:
```bash
# manually specify min coverage
mdca --data='path/to/data.csv' --mode=distribution --min-coverage=0.05  
mdca --data='path/to/data.csv' --mode=distribution --min-target-coverage=0.05  
```

You can also specify columns you want to analysis:
```bash
# if you want to ensure column1, column2, column3 to be uniform distributed
mdca --data='path/to/data.csv' --mode=distribution --column='column1, column2, column3'  
```

After execution finished, you will get results like this:

========== Results of Coverage Increase ============


| Coverage (Baseline, +N%, *X)     | Target Rate(Overall +%N) | Result                                                                                          |
| -------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------- |
| 54.52% ( 8.33%, +46.19%, *6.54 ) | 25.95% ( -5.72%)         | [nationality=Dutch, ind-debateclub=False, ind-entrepeneur_exp=False]                            |
| 62.00% (16.67%, +45.33%, *3.72 ) | 29.35% ( -2.32%)         | [nationality=Dutch, ind-international_exp=False]                                                |
| 41.33% (11.11%, +30.21%, *3.72 ) | 35.63% ( +3.96%)         | [gender=male, nationality=Dutch]                                                                |
| 39.40% (11.11%, +28.29%, *3.55 ) | 20.69% (-10.99%)         | [nationality=Dutch, ind-degree=bachelor]                                                        |
| 30.33% ( 4.17%, +26.16%, *7.28 ) | 26.30% ( -5.38%)         | [ind-debateclub=False, ind-international_exp=False, ind-entrepeneur_exp=False, ind-languages=1] |
| ...                              | ...                      | ...                                                                                             |

In this result, there are three columns: **Coverage (Baseline, +N%, *X)**, **Target Rate(Overall +N%)**, and **Result**.
**Coverage** means the actual proportion of rows of the current result in the total data. **Baseline** means the expected coverage of the current result. __+N%, *X__ means the actual coverage is how much and how many times higher than the baseline coverage.

**Baseline** coverage is calculated by the following formula:

$$
\vec{C} = (column1, column2, ..., columnN) ∈ Columns(Data Table)
$$

$$
Baseline Coverage(\vec{C}) = \frac{1}{Unique Value Combinations(\vec{C})}
$$

For example, there are two values of gender: *male*, *female*, and two values of nationality: *China*, *America*.
The value combinations of $ \vec{C}=(gender, nationality) $ are: {*(male, China), (male, America), (female, China), (female, America)*}.
So the $ Unique Value Combinations(\vec{C}) = 4 $, and $ Baseline Coverage(\vec{C}) = \frac{1}{4} = 0.25 $.
This algorithm indicates that the Baseline Coverage is the proportion of rows of a value combination in case of all the data are ideally uniform distributed.

**Target Rate** means the rate of positive samples in the given value combination. **Result** is the given value combination.
