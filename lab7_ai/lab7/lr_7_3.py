import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

input_file = 'data_clustering.txt'
X = np.loadtxt(input_file, delimiter=',')

bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))
print(f"Оцінена ширина вікна: {bandwidth:.4f}")

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters_ = len(np.unique(labels))

print(f"Оцінена кількість кластерів: {n_clusters_}")
print("Координати центрів кластерів:")
print(cluster_centers)

plt.figure(figsize=(10, 8))
colors = 10 * ['r', 'g', 'b', 'c', 'k', 'y', 'm']

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]] + '.', markersize=10)

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            marker='x', s=250, linewidths=3, color='black', zorder=10, label='Centroids')

plt.title(f'Mean Shift Clustering (Кількість кластерів = {n_clusters_})')
plt.legend()
plt.grid(True)
plt.show()