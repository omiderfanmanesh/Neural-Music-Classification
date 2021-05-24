import numpy as np
import joblib
from sklearn.svm import SVC
from config import Model,Training
import pandas
from sklearn.model_selection import GridSearchCV


def main():

    #Read CSV file (Dataset)
    data_set = pandas.read_csv(Training.DATA_PATH, index_col=False)
    #remove columns genre and id
    data_set=data_set.drop(data_set.columns[[-2,-3]],axis=1)
    #Covert it into Array
    data_set = np.array(data_set)
    #Calculate number of rows and columns of data_set
    number_of_rows, number_of_cols = data_set.shape
    # Creating design matrix X and target vector y
    data_x = data_set[:, :number_of_cols - 1]
    data_y = data_set[:, number_of_cols - 1]

    #rcv = RepeatedKFold(n_splits=10,n_repeats=100,random_state=random_state)
    #print('Number of Splits of X: ', rcv.get_n_splits(data_x), '\n')
    # Performing cross validation on our data
    #model = SVC(C=100, gamma=0.08, random_state=1234)
    model = SVC(random_state=1234)
    # create model, perform Repeated CV and evaluate model
    #scores = cross_val_score(model, data_x, data_y, scoring='accuracy', cv=rcv, n_jobs=-1)
    # report performance
    #print('Using ten-fold cross-validation the accuracy achieved is: %.3f +- %.3f' % (mean(scores), std(scores)))

    #Hyperparameter Tuning
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,0.08,'scale'],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['accuracy']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            model, tuned_parameters,cv=10
        )
        clf.fit(data_x, data_y)

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
        print()

    #save model
    joblib.dump(clf.best_estimator_,Model.NAME)


if __name__ == '__main__':
    main()