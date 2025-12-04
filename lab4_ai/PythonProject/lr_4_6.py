import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 2 * np.sin(X) + np.random.uniform(-0.6, 0.6, (m, 1))


def plot_learning_curves(model, X, y, title, ax, ylim=None):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)

        # Зберігаємо RMSE
        train_errors.append(np.sqrt(mean_squared_error(y_train[:m], y_train_predict)))
        val_errors.append(np.sqrt(mean_squared_error(y_val, y_val_predict)))

    ax.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Навчальний набір")
    ax.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Перевірочний набір")
    ax.set_title(title)
    ax.set_xlabel("Розмір навчального набору")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True)
    if ylim:
        ax.set_ylim(ylim)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# А) Лінійна регресія
linear_reg = LinearRegression()
plot_learning_curves(linear_reg, X, y, "Лінійна регресія (Underfitting)", axes[0], ylim=[0, 3])

# Б) Поліноміальна регресія 10-го степеня
polynomial_regression_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_10, X, y, "Поліном 10-го ступеня (Overfitting)", axes[1], ylim=[0, 3])

# В) Поліноміальна регресія 2-го степеня
polynomial_regression_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_2, X, y, "Поліном 2-го ступеня", axes[2], ylim=[0, 3])

plt.tight_layout()
plt.show()