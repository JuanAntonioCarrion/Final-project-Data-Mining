from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_validate, ShuffleSplit, KFold

class CrossValidation():


    @property
    def classifier(self):
        return self.__classifier
    @classifier.setter
    def classifier(self, c):
        self.__classifier = c

    @property
    def X(self):
        return self.__X

    @X.setter
    def X(self, value):
        self.__X = value

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, value):
        self.__y = value

    def validate(self, num_splits, X, y, clf, var_esc):
        std = 0
        mean = 0

        kf = KFold(n_splits=num_splits)  ##Creamos el particionador para num_splits particiones

        scores = cross_val_score(clf, X, y, cv = num_splits)
        print(var_esc)
        std = scores.std()
        mean = scores.mean()
        print(scores)
        print("La desviacion tipica es:", std,
                  ", y la media: ", mean)
        return std, mean

    def chooseBestPartition(self, num_splits):

        kf = KFold(n_splits=num_splits)  ##Creamos el particionador para num_splits particiones

        i = 0
        min_std = 0
        division = kf.split(self.__X, self.__y)
        '''Split nos crea un generador de indices del tamaño del set 
        para hacer validacion cruzada en k-folds'''
        top_train_split = []
        top_test_split = []

        for train_index, test_index in division:
            print("INDICES TRAIN:", train_index, "INDICES TEST:", test_index)
            X_train, X_test = self.__X[train_index], self.__X[
                test_index]  ##Entrenamos el clasificador con cada una de las posibles particiones
            y_train, y_test = self.__y[train_index], self.__y[test_index]
            clf = self.__classifier
            '''Dentro de la validación cruzada, hacemos OTRA validación cruzada para tener luego una media de 
            la puntuación, y una desviación típica, y así saber qué distribución de los datos elegir '''
            scores = cross_val_score(clf, self.__X, self.__y, cv=5)

            if (i == 0):  ##Esta parte del algoritmo es para decidir cual es la mejor particion (la que tenga menos desviación típica)
                min_std = scores.std()
                top_train_split = train_index
                top_test_split = test_index
            else:
                if (min_std > scores.std()):
                    min_std = scores.std()
                    top_train_split = train_index
                    top_test_split = test_index

            print("Dejando el subconjunto ", i, " de test, tiene una desviacion tipica de:", scores.std(),
                  "y una media de: ", scores.mean())
            i += 1

        return top_train_split, top_test_split  # Por último, el algoritmo devuelve la mejor partición de train y de test