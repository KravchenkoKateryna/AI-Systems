import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import warnings
import os

warnings.filterwarnings("ignore")

def visualize_classifier(classifier, X, y, title=''):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    mesh_step_size = 0.01

    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                                 np.arange(min_y, max_y, mesh_step_size))

    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray, shading='auto', alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.xticks((np.arange(int(min_x), int(max_x), 1.0)))
    plt.yticks((np.arange(int(min_y), int(max_y), 1.0)))

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Classify imbalanced data')
    parser.add_argument('--balance', action='store_true', help="Врахувати дисбаланс класів (balanced weights)")
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    input_file = 'data_imbalance.txt'

    if not os.path.exists(input_file):
        print(f"ПОМИЛКА: Файл '{input_file}' не знайдено.")
        print("Переконайтеся, що ви завантажили файл data_imbalance.txt у папку з проектом.")
        exit()

    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])

    plt.figure(figsize=(10, 6))
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='s', label='Class 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='o', label='Class 1')
    plt.title('Вхідні дані (Input data)')
    plt.legend()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

    if args.balance:
        print("\n--- РЕЖИМ: З балансуванням (class_weight='balanced') ---")
        params['class_weight'] = 'balanced'
    else:
        print("\n--- РЕЖИМ: Без балансування ---")

    classifier = ExtraTreesClassifier(**params)
    classifier.fit(X_train, y_train)

    y_test_pred = classifier.predict(X_test)

    if args.balance:
        title = 'Class boundaries (Balanced)'
    else:
        title = 'Class boundaries (Unbalanced)'

    visualize_classifier(classifier, X_test, y_test, title)
    class_names = ['Class-0', 'Class-1']
    print("\n" + "#" * 40)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("#" * 40 + "\n")

    plt.show()