import datetime
import json
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
from sklearn import covariance, cluster

symbol_file = 'company_symbol_mapping.json'
try:
    with open(symbol_file, 'r') as f:
        symbol_dict = json.loads(f.read())
except FileNotFoundError:
    print(f"Помилка: Файл {symbol_file} не знайдено.")
    sys.exit(1)

symbols, names = np.array(list(symbol_dict.items())).T

print("Завантаження даних котирувань з Yahoo Finance...")
start_date = "2023-01-01"
end_date = "2024-01-01"

quotes = []
valid_symbols = []
valid_names = []

data = yf.download(list(symbols), start=start_date, end=end_date, progress=False)

opening_quotes = data['Open']
closing_quotes = data['Close']
opening_quotes = opening_quotes.dropna(axis=1)
closing_quotes = closing_quotes.dropna(axis=1)

available_symbols = opening_quotes.columns.tolist()
valid_names = [symbol_dict[s] for s in available_symbols]
valid_symbols = available_symbols

open_data = opening_quotes.T.to_numpy()
close_data = closing_quotes.T.to_numpy()

variation = close_data - open_data

std_dev = variation.std(axis=1)
variation /= std_dev[:, np.newaxis]

print(f"Дані підготовлено. Розмірність: {variation.shape} (Компаній x Днів)")

print("Навчання моделі графа (GraphicalLasso)...")
edge_model = covariance.GraphicalLassoCV()

edge_model.fit(variation.T)

print("Кластеризація методом Affinity Propagation...")
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=42)
n_labels = labels.max()

print("\n--- Результати кластеризації фондового ринку ---")
for i in range(n_labels + 1):
    cluster_members_indices = np.where(labels == i)[0]
    cluster_names = [valid_names[idx] for idx in cluster_members_indices]

    if len(cluster_names) > 0:
        print(f"Кластер {i + 1}: {', '.join(cluster_names)}")
