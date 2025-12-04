import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

input_file = 'traffic_data.txt'

if not os.path.exists(input_file):
    print(f"ПОМИЛКА: Файл '{input_file}' не знайдено.")
    exit()

X = []
count = 0
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        items = line.split(',')
        X.append(items)

X = np.array(X)

label_encoders = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if i == len(X[0]) - 1:
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoders.append(le)

X_values = X_encoded[:, :-1].astype(int)
y_values = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_values, y_values, test_size=0.25, random_state=5)

params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)
print("Навчання моделі...")
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (Середня абсолютна похибка): {round(mae, 2)}")

test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']

test_datapoint_encoded = []
try:
    for i, item in enumerate(test_datapoint):
        test_datapoint_encoded.append(label_encoders[i].transform([item])[0])

    test_datapoint_encoded = np.array(test_datapoint_encoded).reshape(1, -1)

    predicted_traffic = regressor.predict(test_datapoint_encoded)[0]

    print("\n--- ПРОГНОЗ ---")
    print(f"Вхідні дані: {test_datapoint}")
    print(f"Прогнозована інтенсивність руху: {int(predicted_traffic)}")

except ValueError as e:
    print("\nПомилка кодування даних. Перевірте, чи значення тестової точки існують у навчальному файлі.")
    print(f"Деталі: {e}")