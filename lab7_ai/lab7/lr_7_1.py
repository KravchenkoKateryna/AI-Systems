import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

input_file = 'data_clustering.txt'
X = np.loadtxt(input_file, delimiter=',')

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)
plt.title('Вхідні дані (Input Data)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig('cluster_input_data.png')
plt.show()

num_clusters = 5

kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
kmeans.fit(X)

step_size = 0.01

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size),
                             np.arange(y_min, y_max, step_size))

output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)

plt.figure(figsize=(10, 8))
plt.clf()

plt.imshow(output, interpolation='nearest',
           extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=80, label='Data points')

cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            marker='x', s=200, linewidths=4, color='black', zorder=10, label='Centroids')

plt.title('Результат кластеризації k-means (межі та центроїди)')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.legend()
plt.savefig('cluster_output_result.png')
plt.show()