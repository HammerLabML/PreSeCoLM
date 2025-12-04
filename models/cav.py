import numpy as np
from sklearn.linear_model import LogisticRegression


class CAV():

    def __init__(self):
        self.clf = LogisticRegression(max_iter=100000, tol=1e-2)  # default lbfgs solver (handles multi-class), l2 loss
        self.multi_clfs = []
        self.multi_label = False

    def fit(self, X, y):
        if len(y.shape) > 1:
            # multi-label
            self.multi_label = True
            for c in range(y.shape[1]):
                clf = LogisticRegression()
                clf.fit(X, y[:,c])
                self.multi_clfs.append(clf)
        else:
            self.clf.fit(X, y)

    def get_concept_vectors(self):
        if self.multi_label:
            cavs = np.vstack([clf.coef_ for clf in self.multi_clfs])
            return cavs
        else:
            return self.clf.coef_

    def get_concept_activations_per_clf(self, clf, X, use_intercept=False):
        if use_intercept:
            # this uses both coefficients and intercept of the classifier
            pred = clf.decision_function(X)
        else:
            # this is how the tensorflow implementation does it
            # https://github.com/tensorflow/tcav/blob/master/tcav/tcav.py l. 136
            pred = np.dot(X, clf.coef_.T)
        
        return pred

    def get_concept_activations(self, X, use_intercept=False):
        if self.multi_label:
            activations = [self.get_concept_activations_per_clf(clf, X, use_intercept) for clf in self.multi_clfs]
            return np.vstack(activations)
        else:
            return self.get_concept_activations_per_clf(self.clf, X, use_intercept)

    def predict(self, X):
        if self.multi_label:
            preds = [clf.predict(X) for clf in self.multi_clfs]
            return np.vstack(preds)
        else:
            return self.clf.predict(X)