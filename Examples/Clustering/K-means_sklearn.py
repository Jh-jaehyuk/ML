import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, centers=4, cluster_std=0.9)
plt.scatter(X[:, 0], X[:, 1], marker='.')
# plt.show()

k_means = KMeans(init='k-means++', n_clusters=4, n_init=12)
k_means.fit(X)

k_means_labels = k_means.labels_
# print(k_means_labels)

k_means_cluster_centers = k_means.cluster_centers_
# print(k_means_cluster_centers)

fig = plt.figure(figsize=(5, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
ax = fig.add_subplot(1, 1, 1)

for k, col in zip(range(4), colors):
    my_members = (k_means_labels == k)
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

ax.set_title('K-Means')
ax.set_xticks(())
ax.set_yticks(())
plt.show()