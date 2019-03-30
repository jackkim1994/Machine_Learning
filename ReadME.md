# Machine Learning Library for R & Python
A platform for Machine Learning resources of both R and Python.
Some of the useful tips and tricks will be added in this repository.
Note that only the Data Preprocessing will be shown in this ReadME.md as an preparation to tackle next chapters.
Other ML tips and tricks will be added and updated on files and available links.


- [Part 1: Data Preprocessing](https://github.com/jackkim1994/Machine_Learning/tree/master/Part%201%20Data%20Preprocessing)
- Part 2: Regression

## 1. Data Preprocessing
Before we dive into Machine Learning, it is a good idea to clean up the data and prepare the training + testing dataset. Below codes are a short introduction to Data Preprocessing using either R or Python.

### Taking Care of Missing Data
**R**
```r
dataset = read.csv('Data.csv')

dataset$Age = ifelse(is.na(dataset$Age), ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary), ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)
```
`ave()` function acts like an `apply()` function that can utilize the function in filling the NA's with values.


**Python**
```python
import pandas as pd
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
```
**Imputer** from `sklearn.preprocessing` package helps changing values of NA's based on the **strategy** I implemented.
After covering missing values, it is necessary to use `fit` and `transform` function to fill up the missing values.


### Encoding Categorical Data
**R**
```r
dataset$Country = factor(dataset$Country, 
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1, 2, 3))
dataset$Purchased <- factor(dataset$Purchased, 
                           levels = c("No", "Yes"),
                           labels = c(0, 1))
```
A common way is using a `factor()` function to categorize factor variables. However, it is good to note that we used `ordered = TRUE` condition if the variable has a hierarchical order.


**Python**
```python
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0 ])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
```
Generally, we use `LabelEncoder` as long as the variable is ordered.
If the variable such as Country is not ordered, we may have to use `OneHotEncoder` to construct an unordered categorical variable. This package will allow to create **dummy variables** as an alternate solution to the problem.

### Splitting the dataset into the Training set and Test set
**R**
```r
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
```
`sample.split` function helps the preparation of splitting the dataset to training and test sets.
Note that Purchased data is made up of categorical variable TRUE/FALSE, so the split is set up as logical variable.

**Python**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```
`train_test_split` package from `sklearn.model_selection` is a common but one of the most important function needed to split the dataset into training and test set. 
This equation is required before creating ML algorithm.
Note that similar to `set.seed()` from R, `random_state = 0` maintains the same output.
