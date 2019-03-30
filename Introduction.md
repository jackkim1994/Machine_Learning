# Machine Learning Library for both R and Python
A platform for Machine Learning resources of both R and Python.
Some of the useful tips and tricks will be added in this repository.

## 1. Data Preprocessing

### Taking Care of Missing Data
**R**
```r
dataset = read.csv('Data.csv')

dataset$Age = ifelse(is.na(dataset$Age), ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary), ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)
```
ave() function acts like an apply() function that can utilize the function in filling the missing values.


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
**Imputer** from `sklearn.preprocessing` package allows the changing values of NA's.
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

**Python**
```Python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0 ])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
```
