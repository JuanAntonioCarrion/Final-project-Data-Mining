from sklearn.preprocessing import LabelEncoder


class Tokenizer:

    def Tokenize(self, X, Y, X_columns, y_columns):
        X_new, X_dec = self.__changeToNumericalValues(X, X_columns)
        y_new, y_dec = self.__changeToNumericalValues(Y, y_columns)
        return X_new, y_new, X_dec, y_dec

    def __changeToNumericalValues(self, df, columns):


        decoder = []
        for i in columns:
            le = LabelEncoder()
            df[i] = le.fit_transform(df[i])
            decoder.append(le.classes_)

        return df, decoder