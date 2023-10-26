import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

iris = load_iris()
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df = pd.DataFrame(data=iris.data, columns=feature_names)
_df = pd.DataFrame(data=iris.data, columns=feature_names)
df['target'] = iris.target

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0)
kmeans.fit(_df)
_df['target'] = iris.target
_df['cluster'] = kmeans.labels_
gmm = GaussianMixture(n_components=3, random_state=0).fit(iris.data)
gmm_cluster_labels = gmm.predict(iris.data)

df['gmm_cluster'] = gmm_cluster_labels
df['target'] = iris.target

kmeans_result = _df.groupby(['target', 'cluster'])['sepal_length'].count()
gmm_result = df.groupby(['target'])['gmm_cluster'].value_counts()
print(kmeans_result)
print(gmm_result)

gmm_pca = PCA(n_components=2)
gmm_transformed = gmm_pca.fit_transform(iris.data)
df['pca_x'] = gmm_transformed[:, 0]
df['pca_y'] = gmm_transformed[:, 1]

gmm_marker0_idx = df[df['gmm_cluster']==0].index
gmm_marker1_idx = df[df['gmm_cluster']==1].index
gmm_marker2_idx = df[df['gmm_cluster']==2].index

kmeans_pca = PCA(n_components=2)
kmeans_transformed = kmeans_pca.fit_transform(iris.data)
_df['pca_x'] = kmeans_transformed[:, 0]
_df['pca_y'] = kmeans_transformed[:, 1]

kmeans_marker0_idx = _df[_df['cluster']==0].index
kmeans_marker1_idx = _df[_df['cluster']==1].index
kmeans_marker2_idx = _df[_df['cluster']==2].index

fig = plt.figure(figsize=(6, 4))
fig.set_facecolor('white')
plt.scatter(x=_df.loc[kmeans_marker0_idx, 'pca_x'], y=_df.loc[kmeans_marker0_idx, 'pca_y'], marker='o')
plt.scatter(x=_df.loc[kmeans_marker1_idx, 'pca_x'], y=_df.loc[kmeans_marker1_idx, 'pca_y'], marker='s')
plt.scatter(x=_df.loc[kmeans_marker2_idx, 'pca_x'], y=_df.loc[kmeans_marker2_idx, 'pca_y'], marker='^')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2', rotation=0)
plt.title('K-Means')

fig = plt.figure(figsize=(6, 4))
fig.set_facecolor('white')
plt.scatter(x=df.loc[gmm_marker0_idx, 'pca_x'], y=df.loc[gmm_marker0_idx, 'pca_y'], marker='o')
plt.scatter(x=df.loc[gmm_marker1_idx, 'pca_x'], y=df.loc[gmm_marker1_idx, 'pca_y'], marker='s')
plt.scatter(x=df.loc[gmm_marker2_idx, 'pca_x'], y=df.loc[gmm_marker2_idx, 'pca_y'], marker='^')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2', rotation=0)
plt.title('GMM')
plt.show()