import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
print("Завантаження даних... (це може зайняти кілька секунд)")
df = pd.read_csv(url)

df = df.dropna(subset=['price'])

df['Price_Category'] = pd.qcut(df['price'], q=3, labels=['Low', 'Medium', 'High'])

print(f"\nРозмір очищеного датасету: {df.shape}")
print("Приклад розподілу цін:")
print(df[['price', 'Price_Category']].head())

le = LabelEncoder()
features_to_encode = ['origin', 'destination', 'train_type', 'train_class', 'fare']

data_encoded = df.copy()

for col in features_to_encode:
    data_encoded[col] = data_encoded[col].fillna('Unknown')
    data_encoded[col] = le.fit_transform(data_encoded[col])

X = data_encoded[['origin', 'destination', 'train_type', 'train_class', 'fare']]
y = data_encoded['Price_Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Результати моделювання ---")
print(f"Точність моделі (Accuracy): {accuracy*100:.2f}%")
print("\nЗвіт класифікації:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Передбачено моделлю')
plt.ylabel('Фактично')
plt.title('Матриця плутанини (Confusion Matrix)')
plt.show()