from __future__ import print_function, division

import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from utils.dataset_helper import pad_along_axis

print(torch.__version__)
print(torchaudio.__version__)

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

plt.ion()

import pathlib

print(pathlib.Path().absolute())


class GTZAN(Dataset):
    def __init__(self, cfg, transforms, training_type=0):

        self.load_from_numpy = cfg.DATALOADER.LOAD_FROM_NUMPY
        if training_type == 0:
            self.np_samples_address = cfg.DATALOADER.NPY_SAMPLES_TRAIN_DATASET_ADDRESS
            self.np_labels_address = cfg.DATALOADER.NPY_LABELS_TRAIN_DATASET_ADDRESS
        elif training_type == 1:
            self.np_samples_address = cfg.DATALOADER.NPY_SAMPLES_TEST_DATASET_ADDRESS
            self.np_labels_address = cfg.DATALOADER.NPY_LABELS_TEST_DATASET_ADDRESS
        elif training_type == 2:
            self.np_samples_address = cfg.DATALOADER.NPY_SAMPLES_VALIDATION_DATASET_ADDRESS
            self.np_labels_address = cfg.DATALOADER.NPY_LABELS_VALIDATION_DATASET_ADDRESS

        self.genre_folder = cfg.DATALOADER.DATASET_ADDRESS
        self.one_hot_encoding = cfg.DATALOADER.ONE_HOT_ENCODING

        if self.load_from_numpy:
            self.samples, self.labels = self.load_from_np()
        else:
            self.samples, self.labels = self.extract_address()
            self.sr = cfg.DATALOADER.SR
            self.n_mels = cfg.DATALOADER.N_MELS
            self.n_fft = cfg.DATALOADER.N_FFT
            self.hop_length = cfg.DATALOADER.HOP_LENGTH

        self.transform = transforms
        self.le = LabelEncoder()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.load_from_numpy:
            label_map = {
                'blues': 0,
                'classical': 1,
                'country': 2,
                'disco': 3,
                'hiphop': 4,
                'jazz': 5,
                'metal': 6,
                'pop': 7,
                'reggae': 8,
                'rock': 9
            }

            # label_map = {
            #     'blues': 0,
            #     'classical': 1,
            #     'disco': 2,
            #     'jazz': 3,
            #     'metal': 4,
            #     'pop': 5,
            #     'rock': 6
            # }
            sample = self.samples[index]
            # sample = add_noise(signal=sample)
            if sample.ndim == 2:
                sample = np.expand_dims(sample, axis=0)
            # if sample.shape[2] != 94:
            #     sample = pad_along_axis(sample, 128, axis=2)
            if sample.shape[2] < 100:
                sample = sample[:, :, :91]
            print(sample.shape)
            sample = torch.from_numpy(sample)
            # sample = rand_aug_sample(sample)
            label = self.labels[index]
            label = label_map[label]
        else:
            address = self.samples[index]
            y, sr = librosa.load(address, sr=self.sr)
            S = librosa.feature.melspectrogram(y, sr=sr,
                                               n_mels=self.n_mels,
                                               n_fft=self.n_fft,
                                               hop_length=self.hop_length)

            sample = librosa.amplitude_to_db(S, ref=1.0)
            # sample = np.expand_dims(sample, axis=0)
            sample = pad_along_axis(sample, 128, axis=2)
            # print(sample.shape)
            sample = torch.from_numpy(sample)

            label = self.labels[index]
            # label = torch.from_numpy(label)
            # print(sample.shape, label)
            if self.transform:
                sample = self.transform(sample)

        return sample, label

    def load_from_np(self):
        return np.load(self.np_samples_address), \
               np.load(self.np_labels_address)

    def extract_address(self):
        label_map = {
            'blues': 0,
            'classical': 1,
            'country': 2,
            'disco': 3,
            'hiphop': 4,
            'jazz': 5,
            'metal': 6,
            'pop': 7,
            'reggae': 8,
            'rock': 9
        }
        labels = []
        address = []
        # extract all genres' folders
        genres = [path for path in os.listdir(self.genre_folder)]
        for genre in genres:
            # e.g. ./data/generes_original/country
            genre_path = os.path.join(self.genre_folder, genre)
            # extract all sounds from genre_path
            songs = os.listdir(genre_path)

            for song in songs:
                song_path = os.path.join(genre_path, song)
                genre_id = label_map[genre]
                # one_hot_targets = torch.eye(10)[genre_id]
                labels.append(genre_id)
                address.append(song_path)

        samples = np.array(address)
        labels = np.array(labels)
        # convert labels to one-hot encoding
        # if self.one_hot_encoding:
        #     labels = OneHotEncoder(sparse=False).fit_transform(labels.reshape((-1,1)))
        # else:
        #     labels = LabelEncoder().fit_transform(labels)

        return samples, labels
