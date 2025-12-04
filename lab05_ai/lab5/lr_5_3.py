import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
import os

input_file = 'data_random_forests.txt'

if not os.path.exists(input_file):
    print(f"ПОМИЛКА: Файл '{input_file}' не знайдено.")
    exit()

data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

parameter_grid = [
    {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
    {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}
]

metrics = ['precision_weighted', 'recall_weighted']

for metric in metrics:
    print("\n" + "#" * 30)
    print(f"Searching optimal parameters for: {metric}")
    print("#" * 30)

    classifier = GridSearchCV(
        ExtraTreesClassifier(random_state=0),
        parameter_grid,
        cv=5,
        scoring=metric
    )

    classifier.fit(X_train, y_train)

    print("\nGrid scores for the parameter grid:")
    results = classifier.cv_results_
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        print(f"  {params} --> {round(mean_score, 3)}")

    print(f"\nBest parameters for {metric}: {classifier.best_params_}")

    y_true, y_pred = y_test, classifier.predict(X_test)
    print("\nPerformance report on test set:")
    print(classification_report(y_true, y_pred))