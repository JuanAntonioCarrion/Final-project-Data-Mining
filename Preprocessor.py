from enum import Enum
import pandas as pd
import numpy as np


class PrepOptions(Enum):
    eliminate_empty = 1
    mean = 2


class Preprocessor():

    def __init__(self, X_columns, y_columns, name):
        self.__X_columns = X_columns
        self.__y_columns = y_columns
        self.__name = name

    def extractData(self, mode_X, mode_y, center_age):
        df = pd.read_csv(self.__name, index_col=False, encoding='latin1')

        df = self.__preprocessData(mode_X, df, self.__X_columns)
        df = self.__preprocessData(mode_y, df, self.__y_columns)

        X = df[self.__X_columns]
        y = df[self.__y_columns]

        if (center_age):
            y = self.__centerAge(y)


        return X, y

    def __preprocessData(self, mode, df, columns):
        for i in columns:
            if mode == PrepOptions.mean:
                if isinstance(df[i][0], int) or isinstance(df[i][0], float):
                    df[i] = df[i].replace(np.NaN, df[i].mean())
                else:
                    most_used_value = df[i].mode()[0]
                    df[i].fillna(most_used_value, inplace=True)
            else:
                df = df.dropna(how='any', subset=[i], inplace=False)
        return df

    def __centerAge(self, df):
        df.loc[df.age < 35, 'age'] = 25
        #df.loc[(df['age'] >= 30) & (df['age'] < 50)] = 40
        df.loc[df.age >= 35, 'age'] = 45
        return df