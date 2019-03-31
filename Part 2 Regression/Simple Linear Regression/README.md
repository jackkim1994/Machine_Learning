# Simplie Linear Regression
The first step of machine learning is understanding the simple linear regression. It is simply a process of finding the line of best
fit in univariate variables (one independent (x) and one dependent variable (y)). 

**R**
```r
# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
```
Similar to previous Part 1, we need a data preprocessing.

```r
# Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)
```
- `lm()` function creates a linear regression model that can be used to predict the test set results.
- `predict()` function predicts a future outcome using a regression model created from the function above.


```r
# Visualising the Training set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')
```
![salary_training_r](https://user-images.githubusercontent.com/42131127/55284993-127d0280-5337-11e9-9cba-c2a96ea955e8.png)

The blue line is constructed from the predictive model the R calculated with the regression formula.

```r
# Visualising the Test set results
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')
```
![salary_test_r](https://user-images.githubusercontent.com/42131127/55285009-4d7f3600-5337-11e9-87d5-765e562e51ec.png)

Using the same predictive regression model (blue line), we test if this same predictive model matches with the test set.
In this case, the model fits the test set fairly well.



**Python**
```python
# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_seelection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
```
Similar process to R, we first create training set and test set.

```python
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```
Next, we introduce a new package `sklearn.linear_model`. It has a function called `LinearRegression` that allows the construction of the linear model.

```python
# Predicting the Test set results
y_pred = regressor.predict(X_test)
```

```python
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```
![salary_tr_python](https://user-images.githubusercontent.com/42131127/55285075-c468fe80-5338-11e9-9fb2-91bc7fcaaf1e.png)


```python
# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```
![salary_test_python](https://user-images.githubusercontent.com/42131127/55285076-cd59d000-5338-11e9-982a-11ee85895610.png)

Note that this predictive model, although uses the same predictive model like the model in R, fits the test set better. Because this is predicting the future outcome, the model will not be always the same for different tests.
