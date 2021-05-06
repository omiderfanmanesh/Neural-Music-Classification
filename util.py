import os
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import librosa
import librosa.display
import dill
import pylab
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')


def create_dataset(genre_folder='./Data/genres_original', save_folder='song_data',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512):
    """This function creates the dataset given a folder
     with the correct structure (project root/Data/genres_original/[genres]/*.wav)
    and saves it to a specified folder."""

    # get list of all artists
    os.makedirs(save_folder, exist_ok=True)
    # print(os.listdir(genre_folder),os.path.isdir('country'))
    # extract all genres' folders
    genres = [path for path in os.listdir(genre_folder)]

    # iterate through all artists, albums, songs and find mel spectrogram
    for genre in tqdm(genres):
        print(genre)
        # e.g. ./Data/generes_original/country
        genre_path = os.path.join(genre_folder, genre)
        # extract all sounds from genre_path
        songs = os.listdir(genre_path)

        for song in tqdm(songs):
            song_path = os.path.join(genre_path, song)

            # Create mel spectrogram and convert it to the log scale
            y, sr = librosa.load(song_path, sr=sr)
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                               n_fft=n_fft,
                                               hop_length=hop_length)
            log_S = librosa.amplitude_to_db(S, ref=1.0)
            target_folder = save_folder + '/' + genre
            os.makedirs(target_folder, exist_ok=True)
            save_name = song.split('.wav')[0] + '.png'
            save_path = target_folder + '/' + save_name
            librosa.display.specshow(log_S)
            pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
            pylab.close()

# if __name__ == '__main__':
#     create_dataset()
