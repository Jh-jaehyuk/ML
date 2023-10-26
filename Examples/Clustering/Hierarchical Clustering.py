import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

iris = load_iris()
labels = pd.DataFrame(iris.target)
labels.columns = ['labels']
df = pd.DataFrame(iris.data)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df = pd.concat([df, labels], axis=1)

mergings = linkage(df, method='complete')

plt.figure(figsize=(5, 4))
dendrogram(mergings,
           labels=labels['labels'].to_numpy(),
           leaf_rotation=90,
           leaf_font_size=10)
# plt.show()

# 3개의 군집으로 나눈 결과
predict = pd.DataFrame(fcluster(mergings, 3, criterion='distance'))
predict.columns = ['predict']
ct = pd.crosstab(predict['predict'], labels['labels'])
print(ct)
