import joblib
import pandas as pd
from config import Model, Test
from sklearn.metrics import classification_report
from Utilities import *


def main():
    data_set1 = pd.read_csv(Test.DATA_PATH, index_col=False)
    # remove columns genre and id
    data_set = data_set1.drop(data_set1.columns[[-2, -3]], axis=1)
    # Covert it into Array
    data_set = np.array(data_set)
    data_set1 = np.array(data_set1)
    # Calculate number of rows and columns of data_set
    number_of_rows, number_of_cols = data_set.shape
    number_of_rows1,number_of_cols1 = data_set1.shape

    # Get X and y data
    X_test = data_set[:, :number_of_cols - 1]
    y_test = data_set1[:, number_of_cols1 - 3]


    #predict genres
    svm = joblib.load(Model.NAME)

    predicted = svm.predict(X_test)

    genre = {"blues": 0, "classical": 1,
              "country": 2, "disco": 3,
              "hiphop": 4, "jazz": 5,
              "metal": 6, "pop": 7,
              "reggae": 8, "rock": 9}

    final_predicted = []

    for predicts in predicted:
        for k,v in genre.items():
            if predicts == v:
                final_predicted.append(k)


    print(classification_report(y_test, final_predicted))


if __name__ == '__main__':
    main()