from sklearn.neural_network import MLPClassifier

X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
clf.fit(X, y)

print(clf.predict([[2., 2.], [-1., -2]]))
print(clf.coefs_)
print([coef.shape for coef in clf.coefs_])
