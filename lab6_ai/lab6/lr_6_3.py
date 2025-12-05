import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

data = {
    'Day': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14'],
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny',
                'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

le_outlook = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()
le_play = LabelEncoder()

df['Outlook_n'] = le_outlook.fit_transform(df['Outlook'])
df['Humidity_n'] = le_humidity.fit_transform(df['Humidity'])
df['Wind_n'] = le_wind.fit_transform(df['Wind'])
y = le_play.fit_transform(df['Play'])

X = df[['Outlook_n', 'Humidity_n', 'Wind_n']]

model = CategoricalNB()
model.fit(X, y)

condition_outlook = 'Overcast'
condition_humidity = 'High'
condition_wind = 'Weak'

print(f"--- Прогноз для умов: {condition_outlook}, {condition_humidity}, {condition_wind} ---")

try:
    query = [[
        le_outlook.transform([condition_outlook])[0],
        le_humidity.transform([condition_humidity])[0],
        le_wind.transform([condition_wind])[0]
    ]]

    prediction_index = model.predict(query)
    prediction_proba = model.predict_proba(query)
    prediction_class = le_play.inverse_transform(prediction_index)

    print(f"\nПрогнозоване рішення: {prediction_class[0]}")
    print(f"Ймовірність 'No' (Гри не буде): {prediction_proba[0][0] * 100:.2f}%")
    print(f"Ймовірність 'Yes' (Гра буде):   {prediction_proba[0][1] * 100:.2f}%")

except ValueError as e:
    print("Помилка: Перевірте правильність написання вхідних даних (регістр має значення).")