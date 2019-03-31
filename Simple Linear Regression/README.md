# Simplie Linear Regression
The first step of machine learning is understanding the simple linear regression. It is simply a process of finding the line of best
fit in univariate variables (one independent (x) and one dependent variable (y)). 

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
