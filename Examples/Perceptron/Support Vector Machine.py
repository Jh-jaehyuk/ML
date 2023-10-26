import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations
from collections import Counter
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

class BinarySVC():
    def __init__(self, kernel=None, coef0=0):
        self.labels_map = None
        self.kernel = kernel
        self.X = None
        self.y = None
        self.alphas = None
        self.w = None
        self.b = None
        self.coef0 = coef0

        if kernel is not None:
            assert kernel in ['poly', 'rbf', 'sigmoid']

    def make_label_map(self, uniq_labels):
        labels_map = list(zip([-1, 1], uniq_labels))
        self.labels_map = labels_map
        return

    def transform_label(self, label, labels_map):
        res = [l[0] for l in labels_map if l[1] == label][0]
        return res

    def inverse_label(self, svm_label, labels_map):
        try:
            res = [l[1] for l in labels_map if l[0] == svm_label][0]
        except:
            print(svm_label)
            print(labels_map)
            raise
        return res

    def get_kernel_val(self, x, y):
        X = self.X
        coef0 = self.coef0
        gamma = 1 / (X.shape[1] * X.var())

        if self.kernel == 'poly':
            return (gamma * np.dot(x, y) + coef0) ** 2

        elif self.kernel == 'rbf':
            return np.exp(-gamma * np.square(np.linalg.norm(x, y)))

        else:
            return np.tanh(gamma * np.dot(x, y) + coef0)

    def fit(self, X, y, C):
        assert C >= 0, 'constant C must be non-negative'
        uniq_labels = np.unique(y)
        assert len(uniq_labels) == 2, 'the number of labels is 2'
        self.make_label_map(uniq_labels)
        self.X = X

        # transfer to -1 or 1
        y = [self.transform_label(label, self.labels_map) for label in y]
        y = np.array(y)

        # formulating standard form
        m, n = X.shape
        y = y.reshape(-1, 1) * 1.
        self.y = y

        if self.kernel is not None:
            Q = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    Q[i][j] = y[i] * y[j] * self.get_kernel_val(X[i], X[j])

        else:
            yX = y * X
            Q = np.dot(yX, yX.T)

        P = cvxopt_matrix(Q)
        q = cvxopt_matrix(-np.ones((m, 1)))
        G = cvxopt_matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
        A = cvxopt_matrix(y.reshape(1, -1))
        b = cvxopt_matrix(np.zeros(1))

        # cvxopt configuration
        cvxopt_solvers.options['show_progress'] = False
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
        S = (alphas > 1e-4).flatten()

        if self.kernel is not None:
            sum_val = 0
            S_index = np.where(S == True)[0]
            for s in S_index:
                temp_vec = np.array([self.get_kernel_val(z, X[s]) for z in X])
                temp_vec = np.expand_dims(temp_vec, axis=1)
                sum_val += np.sum(y[s] - np.sum(y * alphas * temp_vec))
            b = sum_val / len(S)
            self.b = b

        else:
            w = ((y * alphas).T@X).reshape(-1, 1)
            b = np.mean(y[S] - np.dot(X[S], w))
            self.w = w
            self.b = b

        self.alphas = alphas
        return

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        if self.kernel:
            temp_vec = np.array([self.get_kernel_val(x, y) for y in self.X])
            temp_vec = np.expand_dims(temp_vec, axis=1)
            S = (self.alphas > 1e-4).flatten()
            res = np.sign(np.sum(self.y[S] * self.alphas[S] * temp_vec[S]) + self.b)

        else:
            res = np.sign(self.w.T.dot(x) + self.b)

        res = self.inverse_label(res, self.labels_map)
        return res

class mySVM():
    def __init__(self, svm_type='classification', kernel=None, coef0=0):
        assert svm_type in ['classification', 'regression']
        self.svm_type = svm_type
        self.kernel = kernel
        self.X = None
        self.y = None
        self.coef0 = coef0
        self.model_list = None

        if kernel is not None:
            assert kernel in ['poly', 'rbf', 'sigmoid']

    def get_kernel_val(self, x, y):
        X = self.X
        coef0 = self.coef0
        gamma = 1 / (X.shape[1] * X.var())

        if self.kernel == 'poly':
            return (gamma * np.dot(x, y) + coef0) ** 2

        elif self.kernel == 'rbf':
            return np.exp(-gamma * np.square(np.linalg.norm(x, y)))

        else:
            return np.tanh(gamma * np.dot(x, y) + coef0)

    def fit(self, X, y, C, epsilon=0.1):
        if self.svm_type == 'classification':
            self._fit_svc(X, y, C)
        else:
            self._fit_svr(X, y, C, epsilon)

    def _fit_svc(self, X, y, C):
        uniq_labels = np.unique(y)
        label_combinations = list(combinations(uniq_labels, 2))
        model_list = []

        for lc in label_combinations:
            target_idx = np.array([x in lc for x in y])
            y_restricted = y[target_idx]
            X_restricted = X[target_idx]
            clf = BinarySVC(kernel=self.kernel, coef0=self.coef0)
            clf.fit(X_restricted, y_restricted, C)
            model_list.append(clf)
        self.model_list = model_list
        return

    def _fit_svr(self, X, y, C, epsilon):
        assert C >= 0, 'constant C must be non-negative'
        assert epsilon > 0, 'epsilon C must be positive'
        self.X = X
        m, n = X.shape
        y = y.reshape(-1, 1) * 1
        self.y = y

        if self.kernel is not None:
            Q = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    Q[i][j] = self.get_kernel_val(X[i], X[j])

        else:
            Q = X.dot(X.T)

        I = np.eye(m)
        O = np.zeros((m, m))
        sub_Q = np.hstack([I, -I])
        main_Q = sub_Q.T.dot(Q.dot(sub_Q))
        P = cvxopt_matrix(main_Q)
        q = cvxopt_matrix(epsilon * np.ones((2 * m, 1)) - np.vstack([y, -y]))
        G = np.vstack([np.hstack([-I, O]), np.hstack([I, O]), np.hstack([O, -I]), np.hstack([O, I])])
        G = cvxopt_matrix(G)
        h = cvxopt_matrix(np.hstack([np.zeros(m), C * np.ones(m)] * 2))
        A = cvxopt_matrix(np.ones((m, 1)).T.dot(sub_Q))
        b = cvxopt_matrix(np.zeros(1))

        cvxopt_solvers.options['show_progress'] = False
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        sol_root = np.array(sol['x'])
        alphas = sol_root[:m]
        alphas_star = sol_root[m:]

        S = (alphas > 1e-4).flatten()
        if self.kernel is not None:
            sum_val = []
            s_index = np.where(S == True)[0]
            for s in s_index:
                temp_vec = np.array([self.get_kernel_val(z, X[s]) for z in X])
                temp_vec = np.expand_dims(temp_vec, axis=1)
                sum_val.append(-epsilon + np.sum(y[s] - np.sum((alphas * alphas_star) * temp_vec)))
            b = min(sum_val)
            self.b = b

        else:
            w = alphas.T.dot(X) - alphas_star.T.dot(X)
            w = w.reshape(-1, 1)
            b = -epsilon + np.min(y[S] - np.dot(X[S], w))
            self.w = w
            self.b = b
        self.alphas = sol_root
        return

    def predict(self, X):
        if self.svm_type == 'classification':
            model_list = self.model_list
            prediction = [model.predict(X) for model in model_list]
            prediction = [Counter(pred).most_common(1)[0][0] for pred in list(zip(*prediction))]

        else:
            prediction = [self._predict_reg(x) for x in X]
        return prediction

    def _predict_reg(self, x):
        if self.kernel is not None:
            m, _ = self.X.shape
            sol_root = self.alphas
            alphas = sol_root[:m]
            alphas_star = sol_root[m:]

            temp_vec = np.array([self.get_kernel_val(z, x) for z in self.X])
            temp_vec = np.expand_dims(temp_vec, axis=1)
            pred = np.sum((alphas - alphas_star) * temp_vec) + self.b

        else:
            w = self.w
            b = self.b
            pred = w.dot(x) + b
            pred = pred[0]
        return pred


