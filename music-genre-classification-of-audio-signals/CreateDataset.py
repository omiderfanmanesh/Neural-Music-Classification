import os
import numpy as np
import pandas
import sklearn
from Utilities import *
from config import CreateDataset

DATASET_DIR=CreateDataset.Directory
DATASET_NAME = CreateDataset.Name

#read subdirectory of files
def get_subdirectories(directory):
    return[name for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory,name))]


#get array of songs from a specific folder(genre)'''
def get_sample_arrays(folder_name):
    audios=[]
    #return a sorted list of (audio) files
    path_of_audios = librosa.util.find_files(DATASET_DIR + "/" + folder_name)
    for audio in path_of_audios:
        x,sr = librosa.load(audio,sr=SAMPLING_RATE,duration=5.0)
        audios.append(x)
    audios_numpy = np.array(audios)
    return audios_numpy

# #split the dataframe into: train and test dataframes
# def partionDataFrame(df,ratio):
#     df = df.sample(frac=1).reset_index(drop=True)#shuffling rows
#     df = df.sample(frac=1).reset_index(drop=True)#again shuffling
#     size = df["id"].count()
#     limit = int(ratio*size)
#     train_ds = df.loc[0:limit]
#     test_ds =df.loc[limit:size]
#     train_ds.to_csv("C:/Users/mfran/Desktop/GTZAN/content/train.csv",index=False)
#     test_ds.to_csv("C:/Users/mfran/Desktop/GTZAN/content/test.csv",index=False)
#     print("Partioning done")
#

def main():
    labels = []
    dataset_numpy = None
    genres = get_subdirectories(DATASET_DIR)
    for genre in genres:
        print("-------------------["+genre+"]-----------------------")
        signals = get_sample_arrays(genre)
        for signal in signals:
            row = extract_features(signal)
            print(row)
            if dataset_numpy is None:
                dataset_numpy = np.array(row)

            else:
                #Stack arrays in sequence vertically (row wise).
                dataset_numpy = np.vstack((dataset_numpy,(row)))

            labels.append(genre)

    #standardize feature range
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    dataset_numpy = scaler.fit_transform(dataset_numpy)

    #transforming from numpy dataset into a pandas dataframe
    dataset_pandas = pandas.DataFrame(dataset_numpy, columns=CreateDataset.Feature_Names)

    dataset_pandas["genre"] = labels
    dataset_pandas["id"]=dataset_pandas.groupby("genre")["genre"].cumcount()

    dataset_pandas["genre"] =dataset_pandas["genre"].astype('category')
    dataset_pandas["y"] =dataset_pandas["genre"].cat.codes
    dataset_pandas.to_csv(DATASET_NAME,index=False)
    df = pandas.read_csv(CreateDataset.Name, index_col=False)
    #partionDataFrame(df, 0.9)

if __name__ == '__main__':
    main()
    # Read CSV file (Dataset)
    df = pandas.read_csv(CreateDataset.Name, index_col=False)
    #partionDataFrame(df,0.9)