# refereces
# https://github.com/adityajn105/SVM-From-Scratch
# https://github.com/arkm97/svm-from-scratch
# https://github.com/Sohaib1424/Support-Vector-Machine-from-scratch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

from sklearn.model_selection import cross_val_score

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weight = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # y_i(w . x_i - b) >= 1
                condition = y_[idx] * (np.dot(x_i, self.weight) - self.bias) >= 1
                if condition:
                    self.weight -= self.learning_rate * (2 * self.lambda_param * self.weight)
                else:
                    self.weight -= self.learning_rate * (2 * self.lambda_param * self.weight - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.weight) - self.bias
        return np.sign(approx)
    
    def get_params(self, deep=True):
        return {'learning_rate': self.learning_rate, 'lambda_param': self.lambda_param, 'n_iterations': self.n_iterations}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    
if __name__ == "__main__":
    heart_df = pd.read_csv('heart2.csv')

    X = heart_df.drop('target', axis=1)
    X = np.delete(X, (0), axis=0)

    y = heart_df['target']
    y = np.delete(y, (0), axis=0)
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

    clf = SVM()
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    print('My SVM Accuracy: {:.4%}'.format(accuracy))

    sklearn_svm = svm.SVC(C = 10, kernel='linear')
    sklearn_svm.fit(X_train, y_train)
    sklearn_prediction = sklearn_svm.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_prediction)
    print('Sklearn SVM Accuracy: {:.4%}'.format(sklearn_accuracy))


    # using cross validation
    print('using cross validation')
    accuracy = cross_val_score(clf, X, y, cv=10, scoring="accuracy")
    print(accuracy)
    print(accuracy.mean()*100,'%')