from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from Classifier import Classifier

class LinearSupportVectorClassification(Classifier):

    def __init__(self, grid_parameters, cv, name):
        super().__init__(grid_parameters, cv, name)

    def classify(self):
        classifier = LinearSVC()
        clf = GridSearchCV(classifier, self._grid_parameters)
        X = self.cv.X
        Y = self.cv.y
        clf.fit(X, Y)
        super()._results(clf)
