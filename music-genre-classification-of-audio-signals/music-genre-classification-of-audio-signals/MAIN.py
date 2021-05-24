'''
Information about the files:
 -  config.py file includes some properties like dataset directory, test directory and some properties for signal processing and feature extraction.
 -  Utilities.py file includes useful variables and functions shared between multiple files.
 -  CreateDataset.py file is used for feature extraction and creating dataset.
 -  TrainModel.py file is used for creating and training a model.
 -  Classification.py file is for predicting the genres of test music files.
 -  Main.py file runs CreateDataset.py and TrainModel.py and Classification.py sequentially.'''

import CreateDataset
import TrainModel
import Classification

def main():
    CreateDataset.main()
    TrainModel.main()
    Classification.main()

if __name__ == '__main__':
    main()