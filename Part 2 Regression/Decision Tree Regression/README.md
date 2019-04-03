# Decision Tree
Compared to other linear regression, the decision tree is **non-linear** and **non-continuous** regression model that works differently compared to 
other regression models. Simply, it will split the scatter plot into different subgroups by average.

Note that this algorithm is different from Decision Tree Classifier which will be covered later.


![Sample_DT](https://user-images.githubusercontent.com/42131127/55458107-e5cc2380-55a0-11e9-908e-be75e9ed1c5f.png)

This sample decision tree is an example of utilizing the position salaries depending on the average salary grouped by levels.
Sometimes, this method can be a simple process of averaging out the future decision. However, it will not always work because:
- Decision Tree can be extremely sensitive to small perturbations in the data: a slight change can result in a drastically different tree.
- They can easily overfit. 
- They can have problems out-of-sample prediction, or non-smoothing.

Thus, we have to be aware when using the **Decision Tree** Model.

## R
```r
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))
```
When constructing the decision tree, we use a `rpart` package. We have to be aware that we are splitting the data into subgroups,
so we use `minsplit = 1`.

Note that We have to smooth the curve by using the example below. Otherwise, we will not get an ideal step function plot.
```r
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
```

![Rplot](https://user-images.githubusercontent.com/42131127/55458406-bff34e80-55a1-11e9-8633-cc4a261788fc.png)

## Python
```python
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
```
Python introduces a new package called `sklearn.tree` and improt `DecisionTreeRegressor`.
We set `random_state = 0` to not get the same result.
Finally, we have to set higher resolution for a smoothed curve (or step function plot)

```python
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
```

Then, you will get a similar result like R plot.

![pplot](https://user-images.githubusercontent.com/42131127/55458782-c930eb00-55a2-11e9-9f26-e7c462cd3522.png)
