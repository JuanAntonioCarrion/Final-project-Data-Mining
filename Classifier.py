from abc import ABC, abstractmethod


class Classifier(ABC):

    def __init__(self, grid_parameters, cv, name):
        self._grid_parameters = grid_parameters
        self._cv = cv
        self._name = name

    @abstractmethod
    def classify(self): pass

    @property
    def cv(self):
        return self._cv
    @cv.setter
    def cv(self, value):
        self._cv = value

    @property
    def name(self):
        return self._name

    @cv.setter
    def name(self, value):
        self._name = value

    def _results(self, clf):
        print()
        print(self._name)
        print()
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))