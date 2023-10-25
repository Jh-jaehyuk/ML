from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_wine

dataset = load_wine()

print('Feature: ', dataset.feature_names)
print('Labels: ', dataset.target_names)

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4321)

# Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print(accuracy_score(y_test, y_pred))

# Cross validation
scores = cross_val_score(gnb, X, y, cv=5, scoring='accuracy')
print(scores)
print(scores.mean())
