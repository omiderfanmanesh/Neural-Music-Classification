import numpy as np
from numpy import mean,std
#import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from config import Model,CreateDataset
from sklearn.model_selection import cross_val_score,RepeatedKFold,ShuffleSplit
import pandas


def main():

    #Read CSV file (Dataset)
    #data_set1 = pd.read_csv("C:/Users/mfran/Desktop/GTZAN/content/train.csv", index_col=False)
    data_set = pandas.read_csv(CreateDataset.Name, index_col=False)
    #remove columns genre and id
    data_set=data_set.drop(data_set.columns[[-2,-3]],axis=1)
    #data_set = data_set1.drop(data_set1.columns[[-2,-3]],axis=1)
    #print(data_set)
    #Covert it into Array
    data_set = np.array(data_set)
    #Calculate number of rows and columns of data_set
    number_of_rows, number_of_cols = data_set.shape
    #print(number_of_cols)
    # Creating design matrix X and target vector y
    data_x = data_set[:, :number_of_cols - 1]
    # print(data_x)
    data_y = data_set[:, number_of_cols - 1]
    # print(data_y)


    #cv = ShuffleSplit(n_splits=10,test_size=0.1,random_state=1234)
    #scores = cross_val_score(model,data_x,data_y,cv=10,scoring='accuracy')

    # model = RandomForestClassifier(n_estimators=10)
    # model = MLPClassifier(hidden_layer_sizes=(100,))
    # model = KNeighborsClassifier(n_neighbors=Model.NEIGHBOURS_NUMBER)

    random_state=1234
    rcv = RepeatedKFold(n_splits=10,n_repeats=100,random_state=random_state)
    print('Number of Splits of X: ', rcv.get_n_splits(data_x), '\n')
    # Performing cross validation on our data
    model = SVC(C=100, gamma=0.08, random_state=1234)
    # create model, perform Repeated CV and evaluate model
    scores = cross_val_score(model, data_x, data_y, scoring='accuracy', cv=rcv, n_jobs=-1)
    # report performance
    print('Using ten-fold cross-validation the accuracy achieved is: %.3f +- %.3f' % (mean(scores), std(scores)))

    #model.fit(data_x,data_y)
    #print("Using ten-fold crossvalidation %0.2f accuracy was achieved with a standard deviation of %0.2f " % (scores.mean(), scores.std()))

    #joblib.dump(model,Model.NAME)

if __name__ == '__main__':
    main()