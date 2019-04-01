# Before Applying Linear Regression Model
## Five conditions to test on:
1. Linearity: There must be a linear relationship between the dependent variable and the independent variables.
Scatterplots can show whether there is a linear or curvilinear relationship.
2. Homoscedasticity: This assumption states that the variance of error terms is similar across the values of the
independent variables. A plot of standardized residuals versus predicted values can show whether points are
equally distributed across all values of the independent variables.
3. Multivariate Normality: Multiple Linear Regression assumes that the residuals (the differences between the
observed value of the dependent variable y and the predicted value ȳ are normally distributed.
4. Independence of errors: Multiple Linear Regression assumes that the residuals (the differences between the
observed value of the dependent variable y and the predicted value ȳ are independent.
5. Lack of multicollinearity: Multiple Linear Regression assumes that the independent variables are not highly
correlated with each other. This assumption is tested using Variance Inflation Factor (VIF) values.

## Dummy Variable Trap
A special case in multiple linear regression such that one is adding redundant variables to the constant b. When using the dummy variable in linear regression, we will always exclude one dummy variable before constructing the linear model.
