import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

print("Коефіцієнти регресії (Coefficients): \n", regr.coef_)
print("Вільний член (Intercept): \n", regr.intercept_)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("-" * 30)
print(f"R2 score (Коефіцієнт детермінації): {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print("-" * 30)

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0), alpha=0.7, label='Дані')

ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, label='Ідеальна відповідність')

ax.set_xlabel('Виміряно (Реальні дані)')
ax.set_ylabel('Передбачено (Прогноз моделі)')
ax.set_title('Графік: Виміряні vs Передбачені значення')
ax.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()