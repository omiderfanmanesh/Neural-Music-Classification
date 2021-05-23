import joblib
import sklearn
import pandas as pd
from config import Model
import numpy as np
from sklearn.metrics import classification_report

PATH = "C:/Users/mfran/Desktop/GTZAN/content/test.csv"

def main():
    data_set1 = pd.read_csv(PATH, index_col=False)
    # remove columns genre and id
    data_set = data_set1.drop(data_set1.columns[[-2, -1]], axis=1)
    # Covert it into Array
    data_set = np.array(data_set)
    # Calculate number of rows and columns of data_set
    number_of_rows, number_of_cols = data_set.shape

    # Get X and y data
    data_x = data_set[:, :number_of_cols - 1]
    genres_test = data_set[:, number_of_cols - 1]


    #predict genres
    svm = joblib.load(Model.NAME)
    predicted = svm.predict(data_x)
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

    #print(final_predicted)
    accuracy = sklearn.metrics.accuracy_score(genres_test,final_predicted)
    print("Accuracy on test set is {}".format(accuracy))
    confusion_matrix = sklearn.metrics.confusion_matrix(genres_test,final_predicted)
    print(classification_report(genres_test, final_predicted))
    print(confusion_matrix)


if __name__ == '__main__':
    main()