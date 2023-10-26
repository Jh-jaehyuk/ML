import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth

X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.7, random_state=0)

df = pd.DataFrame(data=X, columns=['ftr1', 'ftr2'])
df['target'] = y

best_bandwidth = estimate_bandwidth(X)
ms = MeanShift(bandwidth=best_bandwidth)
cluster_labels = ms.fit_predict(X)
# print(f'cluster labels : {np.unique(cluster_labels)}')

df['meanshift_label'] = cluster_labels
centers = ms.cluster_centers_
unique_labels = np.unique(cluster_labels)
markers = ['o', 's', '^', 'x', '*']

for label in unique_labels:
    temp_label = df[df['meanshift_label'] == label]
    center_xy = centers[label]

    plt.scatter(x=temp_label['ftr1'], y=temp_label['ftr2'], edgecolor='k', marker=markers[label])
    plt.scatter(x=center_xy[0], y=center_xy[1], s=200, color='gray', alpha=0.9, marker=markers[label])
    plt.scatter(x=center_xy[0], y=center_xy[1], s=70, color='k', edgecolors='k', marker=f'${label}$')

print(df.groupby('target')['meanshift_label'].value_counts())
plt.show()