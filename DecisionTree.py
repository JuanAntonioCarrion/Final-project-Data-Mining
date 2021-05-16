from Classifier import Classifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

class DecisionTree(Classifier):

    def __init__(self, grid_parameters, cv, name):
        super().__init__(grid_parameters, cv, name)

    def classify(self):
        classifier = DecisionTreeClassifier()
        clf = GridSearchCV(classifier, self._grid_parameters)
        X = self.cv.X
        Y = self.cv.y
        clf.fit(X, Y)
        super()._results(clf)