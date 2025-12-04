import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

input_file = 'data_multivar_regr.txt'

try:
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
except OSError:
    print(f"Помилка: Файл '{input_file}' не знайдено.")
    exit()

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)

y_test_pred = linear_regressor.predict(X_test)

print("Linear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.transform(datapoint)

print("\n--- Prediction for datapoint [7.75, 6.35, 5.56] ---")
print("Expected value (approx): 41.35")

linear_prediction = linear_regressor.predict(datapoint)[0]
poly_prediction = poly_linear_model.predict(poly_datapoint)[0]

print(f"Linear regression prediction: {linear_prediction:.2f}")
print(f"Polynomial regression prediction: {poly_prediction:.2f}")

print("\n--- Висновок ---")
if abs(41.35 - poly_prediction) < abs(41.35 - linear_prediction):
    print("Поліноміальна регресія дала точніший результат.")
else:
    print("Лінійна регресія дала точніший результат.")