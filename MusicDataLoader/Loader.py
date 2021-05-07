from __future__ import print_function, division
import os
from sklearn.preprocessing import OneHotEncoder
import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as AUF
import torch.nn.functional as NNF
import torchaudio.transforms as T

import librosa

print(torch.__version__)
print(torchaudio.__version__)

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

plt.ion()


class GTZANDataset(Dataset):
    def __init__(self,
                 genre_folder='./Data/genres_original',
                 one_hot_encoding=True,
                 sr=16000, n_mels=128,
                 n_fft=2048, hop_length=512):
        self.genre_folder = genre_folder
        self.one_hot_encoding = one_hot_encoding
        self.audio_address, self.labels = self.extract_address()
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
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
        sample = torch.from_numpy(sample)

        label = self.labels[index]
        label = torch.from_numpy(label)

        return sample, label

    def extract_address(self):
        labels = []
        address = []
        # extract all genres' folders
        genres = [path for path in os.listdir(self.genre_folder)]
        for genre in genres:
            # e.g. ./Data/generes_original/country
            genre_path = os.path.join(self.genre_folder, genre)
            # extract all sounds from genre_path
            songs = os.listdir(genre_path)

            for song in songs:
                song_path = os.path.join(genre_path, song)
                labels.append(genre)
                address.append(song_path)

        samples = np.array(address)
        labels = np.array(labels).reshape((-1,1))
        # convert labels to one-hot encoding
        if self.one_hot_encoding:
            labels = OneHotEncoder(sparse=False).fit_transform(labels)

        return samples, labels
