# encoding: utf-8

from torch.utils import data
from torch.utils.data import random_split
from torchvision import datasets
import numpy as np
import torch

from .dataset.GTZAN import GTZANDataset
from .transforms import build_transforms

import sys
sys.path.insert(0, 'C:\\Users\\Giovanni Calleris\\Desktop\\Phyton\\PyCharm\\NeuralMusicClassification')
sys.path.append('.')
from NeuralMusicClassification.utils.spec_augment import freq_mask



from NeuralMusicClassification.config import cfg



def npy_loader(path):

    """"

    Loads the numpy array to the training set

    It randomly pick a frequency range
    It shifts the frequency down randomly
    It inverts randomly the first and the second half of the clip along the time axis

    """
    counter_of_slices = 5
    slice1 = np.random.randint(2, counter_of_slices)
    pitch_shift = np.random.randint(-6,0)
    # song_time = np.random.randint(0, 2)
    song_time = 0
    song_lenght_size = cfg.DATALOADER.SONG_LENGTH
    song_freq_size = 128
    array_to_import = np.load(path)
    step = 12
    x = array_to_import[ (slice1 * step - pitch_shift):(slice1 * step + cfg.MODEL.FREQ_RANGE - pitch_shift), :]
    array_of_zeros = np.ones((step, x.shape[1]))*(-32)
    counter = 2
    counter_fix = 2
    list_x = []
    while counter < counter_of_slices:
        if counter == slice1:
            list_x.append(x)
        else:
            list_x.append(array_of_zeros)
        counter += 1

    array_to_import = np.concatenate(list_x, axis=0)
    # array_to_import = np.reshape(array_to_import, (cfg.MODEL.FREQ_RANGE,song_lenght_size))
    if array_to_import.shape == (cfg.MODEL.FREQ_RANGE+(counter_of_slices-counter_fix-1)*step, song_lenght_size):

        if song_time == 1:
            # array_to_import = np.flip(array_to_import, axis=-1).copy()
            a = array_to_import[ :, :47]
            b = array_to_import[ :, 47:]
            array_to_import = np.stack([b, a], axis=-1)
            array_to_import = np.reshape(array_to_import, (cfg.MODEL.FREQ_RANGE, song_lenght_size))
        # array_to_import = np.stack([np.real(array_to_import), np.imag(array_to_import)], axis=1)
        # #
        # resize_shape = list(array_to_import.shape)[0] * list(array_to_import.shape)[1]
        # array_to_import = np.reshape(array_to_import,
        #                        [resize_shape, list(array_to_import.shape)[2], list(array_to_import.shape)[3]])
        # array_to_import1 = (np.round(array_to_import[:,0:92]-array_to_import[:,1:93])==0) + 1
        # array_to_import2 = (np.round(array_to_import[:,1:93]-array_to_import[:,2:song_lenght_size])==0) + 1
        # array_to_import = (np.round(array_to_import1 - array_to_import2)==0) + 1
        # array_to_import = np.round(array_to_import[:,0:92]-array_to_import[:,1:93])==0
        # array_to_import_delta = array_to_import[:, 0:92] - array_to_import[:, 1:93]
        # array_to_import = np.stack([array_to_import[:,0:92], array_to_import_delta], axis=0)

        array_to_import = np.reshape(array_to_import, (1, cfg.MODEL.FREQ_RANGE+(counter_of_slices-counter_fix-1)*step, song_lenght_size))

        tensor = torch.from_numpy(array_to_import).type(torch.FloatTensor)
        # if np.random.randn() > 0:
        #     tensor = freq_mask(tensor, F=5, num_masks=1, replace_with_zero=True)
        return [tensor, slice1-counter_fix, song_time]
    else:
        return [torch.zeros([1, cfg.MODEL.FREQ_RANGE+(counter_of_slices-counter_fix-1)*step, song_lenght_size]), 0, song_time]
        # return torch.zeros([4, 32, song_lenght_size])

def npy_loader_test(path):

    """"

    Loads the numpy array to the training set

    """

    song_time = 0
    array_to_import = np.load(path)[:,:]
    song_lenght_size = cfg.DATALOADER.SONG_LENGTH
    song_freq_size = 47

    # array_to_import = np.stack([np.real(array_to_import), np.imag(array_to_import)], axis=1)
    #
    # resize_shape = list(array_to_import.shape)[0] * list(array_to_import.shape)[1]
    # array_to_import = np.reshape(array_to_import,
    #                              [resize_shape, list(array_to_import.shape)[2], list(array_to_import.shape)[3]])


    if array_to_import.shape == (song_freq_size, song_lenght_size):
        # array_to_import1 = (np.round(array_to_import[:, 0:92] - array_to_import[:, 1:93]) == 0) + 1
        # array_to_import2 = (np.round(array_to_import[:, 1:93] - array_to_import[:, 2:song_lenght_size]) == 0) + 1
        # array_to_import = (np.round(array_to_import1 - array_to_import2) == 0) + 1
        # array_to_import_delta = array_to_import[:, 0:92] - array_to_import[:, 1:93]
        #
        # array_to_import = np.stack([array_to_import[:,0:92], array_to_import_delta], axis=0)

        array_to_import = np.reshape(array_to_import, (1, song_freq_size, song_lenght_size))

        tensor = torch.from_numpy(array_to_import).type(torch.FloatTensor)
        return [tensor, 0, song_time]
    else:
        return [torch.zeros([1, song_freq_size, song_lenght_size]), 0, song_time]
        # return torch.zeros([4, song_freq_size, song_lenght_size])

def npy_loader_based_model(path):

    """"

    Loads the numpy array to the training set

    """

    array_to_import = np.load(path)
    song_lenght_size = cfg.DATALOADER.SONG_LENGTH
    song_freq_size = 128
    # std = np.std(array_to_import, axis=1)
    # mean = np.mean(array_to_import, axis=1)
    # array_to_import = np.stack([mean, std], axis=1)
    # array_to
    if array_to_import.shape == (song_freq_size, song_lenght_size):
        array_to_import = np.reshape(array_to_import, (1, song_freq_size, song_lenght_size))

        tensor = torch.from_numpy(array_to_import).type(torch.FloatTensor)
        return tensor
    else:
        return torch.zeros([1, song_freq_size, song_lenght_size])

def build_dataset(cfg, transforms):
    train_dataset, test_dataset, validation_dataset  = GTZANDataset(cfg=cfg, transforms=transforms)
    return train_dataset, test_dataset, validation_dataset


def make_data_loader(cfg, test_size=0.10, validation_size=0.10, shuffle=True):
    # transforms = build_transforms(cfg)
    # train_dataset, test_dataset, validation_dataset = build_dataset(cfg=cfg, transforms=transforms)
    #
    # dataset_size = len(datasets)
    # test_size = int(test_size * dataset_size)
    # train_size = dataset_size - test_size
    #
    # train_dataset, test_dataset = random_split(datasets,
    #                                            [train_size, test_size])
    #
    # validation_size = int(validation_size * train_size)
    # train_size = train_size - validation_size
    #
    # train_dataset, validation_dataset = random_split(train_dataset,
    #                                                  [train_size, validation_size])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.TEST.IMS_PER_BATCH

    # train_data_loader = data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    # )
    #
    # test_data_loader = data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    # )
    #
    # validation_data_loader = data.DataLoader(
    #     validation_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    # )
    if cfg.MODEL.BASED_MODEL:
        print("Preparing loaders for paper's based model")
        loader_training = npy_loader_based_model
        loader_test = npy_loader_based_model

    else:
        print("Preparing loaders for model extension")
        loader_training = npy_loader
        loader_test = npy_loader_test

    train_data = datasets.DatasetFolder(cfg.DATALOADER.NPY_SAMPLES_TRAINING_DATASET_ADDRESS,
                           extensions="npy",
                           loader=loader_training)


    test_data = datasets.DatasetFolder(cfg.DATALOADER.NPY_SAMPLES_TEST_DATASET_ADDRESS,
                                       extensions="npy",
                                       loader=loader_test)

    validation_data = datasets.DatasetFolder(cfg.DATALOADER.NPY_SAMPLES_VALIDATION_DATASET_ADDRESS,
                                       extensions="npy",
                                       loader=loader_test)

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)  # , num_workers=4
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=cfg.SOLVER.IMS_PER_BATCH_VAL_AND_TEST, shuffle=False, drop_last=False)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.SOLVER.IMS_PER_BATCH_VAL_AND_TEST, shuffle=False, drop_last=False)

    return train_data_loader, test_data_loader, validation_data_loader
