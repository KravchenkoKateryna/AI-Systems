import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import os

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
    parser = argparse.ArgumentParser(description='Classify data using Ensemble Learning techniques')

    parser.add_argument('--classifier-type', dest='classifier_type',
                        required=False, default='rf', choices=['rf', 'erf'],
                        help="Type of classifier to use; can be either 'rf' or 'erf'")
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type

    input_file = 'data_random_forests.txt'

    if not os.path.exists(input_file):
        print(f"\nПОМИЛКА: Файл '{input_file}' не знайдено!")
        print("Переконайтеся, що файл з даними лежить у тій самій папці, що і цей скрипт.")
        exit()

    try:
        data = np.loadtxt(input_file, delimiter=',')
        X, y = data[:, :-1], data[:, -1]
    except Exception as e:
        print(f"Помилка при читанні файлу: {e}")
        exit()

    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])
    class_2 = np.array(X[y == 2])

    plt.figure(figsize=(10, 6))
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='s', label='Class 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='o', label='Class 1')
    plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='^', label='Class 2')
    plt.title('Вхідні дані (Input data)')
    plt.legend()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

    if classifier_type == 'rf':
        print("\n--- Використовується Random Forest (rf) ---")
        classifier = RandomForestClassifier(**params)
        title_prefix = "Random Forest"
    else:
        print("\n--- Використовується Extremely Random Forests (erf) ---")
        classifier = ExtraTreesClassifier(**params)
        title_prefix = "Extremely Random Forests"

    classifier.fit(X_train, y_train)

    visualize_classifier(classifier, X_train, y_train, f'{title_prefix}: Training dataset')

    y_test_pred = classifier.predict(X_test)

    visualize_classifier(classifier, X_test, y_test, f'{title_prefix}: Test dataset (Границі)')

    class_names = ['Class-0', 'Class-1', 'Class-2']
    print("\n" + "#" * 40)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("#" * 40 + "\n")

    test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])

    print("\nConfidence measure (Рівні довіри):")
    for datapoint in test_datapoints:
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = 'Class-' + str(np.argmax(probabilities))
        print(f'Datapoint: {datapoint} -> Predicted: {predicted_class}')
        print(f'   Probabilities: {probabilities}')

    visualize_classifier(classifier, test_datapoints, [0] * len(test_datapoints),
                         f'{title_prefix}: Test Datapoints Confidence')

    plt.show()