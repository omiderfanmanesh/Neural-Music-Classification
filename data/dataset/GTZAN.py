from __future__ import print_function, division

import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from utils.util import pad_along_axis

print(torch.__version__)
print(torchaudio.__version__)

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

plt.ion()

import pathlib

print(pathlib.Path().absolute())


class GTZANDataset(Dataset):
    def __init__(self,
                 genre_folder='/home/omid/OMID/projects/python/mldl/NeuralMusicClassification/data/dataset/genres_original',
                 one_hot_encoding=False,
                 sr=16000, n_mels=128,
                 n_fft=2048, hop_length=512,
                 transform=None):

        self.genre_folder = genre_folder
        self.one_hot_encoding = one_hot_encoding
        self.audio_address, self.labels = self.extract_address()
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.transform = transform
        self.le = LabelEncoder()
        self.hop_length = hop_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        address = self.audio_address[index]
        y, sr = librosa.load(address, sr=self.sr)
        S = librosa.feature.melspectrogram(y, sr=sr,
                                           n_mels=self.n_mels,
                                           n_fft=self.n_fft,
                                           hop_length=self.hop_length)

        sample = librosa.amplitude_to_db(S, ref=1.0)
        sample = np.expand_dims(sample, axis=0)
        sample = pad_along_axis(sample, 1024, axis=2)
        # print(sample.shape)
        sample = torch.from_numpy(sample)

        label = self.labels[index]
        # label = torch.from_numpy(label)
        print(sample.shape, label)
        if self.transform:
            sample = self.transform(sample)
        return sample, label

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
        #     labels = OneHotEncoder(sparse=False).fit_transform(labels)
        # else:
        #     labels = LabelEncoder().fit_transform(labels)

        return samples, labels
