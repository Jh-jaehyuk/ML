import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load Dataset
X, Y = load_breast_cancer(return_X_y=True)
X = np.array(X)
Y = np.array(Y)

print(len(X))
print(len(X[0]))

# Classify train set, test set
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=321)

print(len(train_Y))
print(len(train_Y) - sum(train_Y)) # 클래스가 0인 train set
print(sum(train_Y)) # 클래스가 1인 train set
print(len(test_Y))
print(len(test_Y) - sum(test_Y)) # 클래스가 0인 test set
print(sum(test_Y)) # 클래스가 1인 test set

# Initialize Decision Tree model
model = DecisionTreeClassifier()
model.fit(train_X, train_Y)

y_pred_train = model.predict(train_X)
y_pred_test = model.predict(test_X)

# Calculate Confusion Matrix
cm_train = confusion_matrix(train_Y, y_pred_train)
cm_test = confusion_matrix(test_Y, y_pred_test)
print(cm_train)
print(cm_test)

# Visualize Confusion Matrix
fig = plt.figure(figsize=(5, 5))
ax = sns.heatmap(cm_train, annot=True)
ax.set(title='Train Confusion Matrix',
       ylabel='True Label',
       xlabel='Predicted Label')
fig = plt.figure(figsize=(5, 5))
ax = sns.heatmap(cm_test, annot=True)
ax.set(title='Test Confusion Matrix',
       ylabel='True Label',
       xlabel='Predicted Label')
plt.show()

# Print Accuracy
acc_train = model.score(train_X, train_Y)
acc_test = model.score(test_X, test_Y)
print(acc_train)
print(acc_test)

# Print Precision
precision_train = precision_score(train_Y, y_pred_train)
precision_test = precision_score(test_Y, y_pred_test)
print(precision_train)
print(precision_test)

# Print Recall
recall_train = recall_score(train_Y, y_pred_train)
recall_test = recall_score(test_Y, y_pred_test)
print(recall_train)
print(recall_test)
