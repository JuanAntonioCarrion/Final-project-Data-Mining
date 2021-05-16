from sklearn.model_selection import train_test_split

from CrossValidation import CrossValidation
from DecisionTree import DecisionTree
from LinearSupportVectorClassification import LinearSupportVectorClassification
from Preprocessor import Preprocessor as Prep, PrepOptions
from SGDClassifier import SGDClf
from Tokenizer import Tokenizer

# Press the green button in the gutter to run the script.
from XGBoost import XGBoost

if __name__ == '__main__':

    X_columns = ['manner_of_death', 'armed', 'gender', 'race', 'signs_of_mental_illness', 'flee']
    y_columns = ['age']
    filename = "PoliceKillingsUS.csv"

    Preprocessor = Prep(X_columns,y_columns, filename)
    X_option = PrepOptions.mean
    y_option = PrepOptions.eliminate_empty
    X, y = Preprocessor.extractData(X_option, y_option, center_age=True)

    tokenizer = Tokenizer()
    X, y, X_dec, y_dec = tokenizer.Tokenize(X,y, X_columns, y_columns)

    pipeline = []

    cv = CrossValidation()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    cv.X = X_train
    cv.y = y_train

    parameters = {'max_iter': (1,2,5,10,20,50,100,200,500,1000), 'loss':('hinge','log', 'modified_huber', 'squared_hinge', 'perceptron'), 'random_state':(42,)}
    pipeline.append(SGDClf(parameters, cv, "SGDClf"))

    #parameters = {'criterion': ('gini', 'entropy'), 'max_depth': (None, 2, 3, 5, 10, 12), 'min_samples_split': (2, 5, 10, 100),
    #             'min_samples_leaf': (2, 5, 10), 'random_state':(42,)}
    #pipeline.append(DecisionTree(parameters, cv, "DECISION TREE"))

    #parameters = {'booster': ('gbtree', 'gblinear', 'dart'), 'n_estimators':  (100, 200), 'learning_state': (0.001, 0.01, 0.1, 0.2, 0.5)}
    #pipeline.append(XGBoost(parameters, cv, "XGBOOST"))

    #parameters = {'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10], 'max_iter': [1, 5, 30, 40, 50, 1000],
    #              'loss': ['hinge', 'squared_hinge'], 'random_state': [42]}
    #pipeline.append(LinearSupportVectorClassification(parameters, cv, "SVM"))

    for clf in pipeline:
        clf.classify()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/