import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

np.random.seed(42)
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 2 * np.sin(X) + np.random.uniform(-0.6, 0.6, (m, 1))

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

X_new = np.linspace(-3, 3, 100).reshape(-1, 1)
X_new_poly = poly_features.transform(X_new)
y_poly_pred = poly_reg.predict(X_new_poly)

print("--- Лінійна регресія ---")
print(f"R2 Score: {r2_score(y, y_lin_pred):.4f}")
print(f"Рівняння: y = {lin_reg.coef_[0][0]:.2f}x + ({lin_reg.intercept_[0]:.2f})")

print("\n--- Поліноміальна регресія (degree=3) ---")
print(f"R2 Score: {r2_score(y, poly_reg.predict(X_poly)):.4f}")
coefs = poly_reg.coef_[0]
intercept = poly_reg.intercept_[0]
print(f"Коефіцієнти: {coefs}")
print(f"Рівняння: y = {coefs[2]:.2f}x^3 + {coefs[1]:.2f}x^2 + {coefs[0]:.2f}x + ({intercept:.2f})")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Дані (2*sin(X) + шум)')
plt.plot(X, y_lin_pred, color='red', linestyle='--', linewidth=2, label='Лінійна регресія')
plt.plot(X_new, y_poly_pred, color='green', linewidth=3, label='Поліноміальна регресія (deg=3)')

plt.title('Апроксимація функції y = 2sin(x)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()