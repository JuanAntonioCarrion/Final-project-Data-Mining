from sklearn.model_selection import GridSearchCV
from Classifier import Classifier
from sklearn.linear_model import SGDClassifier


class SGDClf(Classifier):

    def __init__(self, grid_parameters, cv, name):
        super().__init__(grid_parameters, cv, name)

    def classify(self):
        sgd_clf = SGDClassifier()
        clf = GridSearchCV(sgd_clf, self._grid_parameters)
        X = self.cv.X
        Y = self.cv.y
        clf.fit(X, Y)
        super()._results(clf)