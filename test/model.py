# from numpy.lib.npyio import load
# import numpy as np
# from sklearn.linear_model import SGDRegressor
# from sklearn.datasets import load_iris

# from regresspy import Regression
# from regresspy import rmse


# iris = load_iris()
# # We will use sepal length to predict sepal width
# X = iris.data[:, 0].reshape(-1, 1)
# Y = iris.data[:, 1].reshape(-1, 1)

# #TODO Perform a linear regression using sklearn and calculate training rmse.
# # Use the SGDRegressor and only select set learning rate and epochs.


# #TODO Perform a linear regression using your code and calculate training rmse.

from numpy.lib.npyio import load
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_iris

from regresspy.regression import Regression
from regresspy.loss import rmse


iris = load_iris()
# We will use sepal length to predict sepal width
X = iris.data[:, 0].reshape(-1, 1)
Y = iris.data[:, 1]
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
print(X_test.shape) 
print(y_test.shape)
print(X_train.shape)
print(y_train.shape)
#TODO Perform a linear regression using sklearn and calculate training rmse.
# Use the SGDRegressor and only select set learning rate and epochs.
model = SGDRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squard_error(y_test_, y_pred)
prev_rmse = math. sqrt(mse)
print("Previous model " + str(prev_rmse))

#TODO Perform a linear regression using your code and calculate training rmse.

my_model = Regression()
my_model.fit(X_train, y_train)
print(my_model._weights['W'].shape)

my_model_y_pred = my_model.predict(X_test)
print(y_test.shape)
print(my_model_y_pred.shape)
my_model_rmse = rmse(y_test_,my_model_y_pred)
print(my_model_rmse)