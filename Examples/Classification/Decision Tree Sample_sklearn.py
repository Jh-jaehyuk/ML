import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Dataset load
X, Y = load_iris(return_X_y= True)

# Convert to Dataframe
df = pd.DataFrame(X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
df['class'] = Y

X = df.drop(columns=['class'])
Y = df['class']

# Classify train set, test set
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize Decision Tree model
model = DecisionTreeClassifier()

# Learn defined model
model.fit(train_X, train_Y)

# Visualize result
fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(model, feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'],
                   class_names=['setosa', 'versicolor', 'virginica'],
                   filled=True)

# Predict result
pred_X = model.predict(test_X)
print(pred_X)
