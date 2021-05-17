import os

import dill
import librosa
import librosa.display
import librosa.display
import numpy as np
import torch
from numpy.random import RandomState
from tqdm import tqdm

print(torch.version.__version__)
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from numpy.random import RandomState

np.random.seed(42)

from pydub import AudioSegment
from pydub.utils import make_chunks


# matplotlib.use('Agg') # No pictures displayed


def create_numpy_dataset(nb_classes=10,
                         slice_length=911,
                         artist_folder='artists',
                         song_folder='np_data',
                         album_split=True,
                         random_states=42
                         ):
    print("Loading dataset...")

    if not album_split:
        # song split
        Y_train, X_train, S_train, Y_test, X_test, S_test, \
        Y_val, X_val, S_val = \
            load_dataset_song_split(song_folder_name=song_folder,
                                    artist_folder=artist_folder,
                                    nb_classes=nb_classes,
                                    random_state=random_states)
    else:
        Y_train, X_train, S_train, Y_test, X_test, S_test, \
        Y_val, X_val, S_val = \
            load_dataset_album_split(song_folder_name=song_folder,
                                     artist_folder=artist_folder,
                                     nb_classes=nb_classes,
                                     random_state=random_states)

    print("Loaded and split dataset. Slicing songs...")

    # Create slices out of the songs
    X_train, Y_train, S_train = slice_songs(X_train, Y_train, S_train,
                                            length=slice_length)
    X_val, Y_val, S_val = slice_songs(X_val, Y_val, S_val,
                                      length=slice_length)
    X_test, Y_test, S_test = slice_songs(X_test, Y_test, S_test,
                                         length=slice_length)

    print("Training set label counts:", np.unique(Y_train, return_counts=True))

    # Encode the target vectors into one-hot encoded vectors
    Y_train, le, enc = encode_labels(Y_train)
    Y_test, le, enc = encode_labels(Y_test, le, enc)
    Y_val, le, enc = encode_labels(Y_val, le, enc)

    # Reshape data as 2d convolutional tensor shape
    X_train = X_train.reshape(X_train.shape + (1,))
    X_val = X_val.reshape(X_val.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))


def load_dataset(song_folder_name='np_data',
                 artist_folder='artists',
                 nb_classes=10, random_state=42):
    """This function loads the dataset based on a location;
     it returns a list of spectrograms
     and their corresponding artists/song names"""

    # Get all songs saved as numpy arrays in the given folder
    song_list = os.listdir(song_folder_name)

    # Load the list of artists
    artist_list = os.listdir(artist_folder)

    # select the appropriate number of classes
    prng = RandomState(random_state)
    artists = prng.choice(artist_list, size=nb_classes, replace=False)

    # Create empty lists
    artist = []
    spectrogram = []
    song_name = []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)
        if loaded_song[0] in artists:
            artist.append(loaded_song[0])
            spectrogram.append(loaded_song[1])
            song_name.append(loaded_song[2])

    return artist, spectrogram, song_name


def encode_labels(Y, le=None, enc=None):
    """Encodes target variables into numbers and then one hot encodings"""

    # initialize encoders
    N = Y.shape[0]

    # Encode the labels
    if le is None:
        le = preprocessing.LabelEncoder()
        Y_le = le.fit_transform(Y).reshape(N, 1)
    else:
        Y_le = le.transform(Y).reshape(N, 1)

    # convert into one hot encoding
    if enc is None:
        enc = preprocessing.OneHotEncoder()
        Y_enc = enc.fit_transform(Y_le).toarray()
    else:
        Y_enc = enc.transform(Y_le).toarray()

    # return encoders to re-use on other data
    return Y_enc, le, enc


def load_dataset_album_split(song_folder_name='np_data',
                             artist_folder='artists',
                             nb_classes=20, random_state=42):
    """ This function loads a dataset and splits it on an album level"""
    song_list = os.listdir(song_folder_name)

    # Load the list of artists
    artist_list = os.listdir(artist_folder)

    train_albums = []
    test_albums = []
    val_albums = []
    random.seed(random_state)
    for artist in os.listdir(artist_folder):
        albums = os.listdir(os.path.join(artist_folder, artist))
        random.shuffle(albums)
        test_albums.append(artist + '_%%-%%_' + albums.pop(0))
        val_albums.append(artist + '_%%-%%_' + albums.pop(0))
        train_albums.extend([artist + '_%%-%%_' + album for album in albums])

    # select the appropriate number of classes
    prng = RandomState(random_state)
    artists = prng.choice(artist_list, size=nb_classes, replace=False)

    # Create empty lists
    Y_train, Y_test, Y_val = [], [], []
    X_train, X_test, X_val = [], [], []
    S_train, S_test, S_val = [], [], []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)
        artist, album, song_name = song.split('_%%-%%_')
        artist_album = artist + '_%%-%%_' + album

        if loaded_song[0] in artists:
            if artist_album in train_albums:
                Y_train.append(loaded_song[0])
                X_train.append(loaded_song[1])
                S_train.append(loaded_song[2])
            elif artist_album in test_albums:
                Y_test.append(loaded_song[0])
                X_test.append(loaded_song[1])
                S_test.append(loaded_song[2])
            elif artist_album in val_albums:
                Y_val.append(loaded_song[0])
                X_val.append(loaded_song[1])
                S_val.append(loaded_song[2])

    return Y_train, X_train, S_train, \
           Y_test, X_test, S_test, \
           Y_val, X_val, S_val


def load_dataset_song_split(song_folder_name='np_data',
                            artist_folder='artists',
                            nb_classes=20,
                            test_split_size=0.1,
                            validation_split_size=0.1,
                            random_state=42):
    Y, X, S = load_dataset(song_folder_name=song_folder_name,
                           artist_folder=artist_folder,
                           nb_classes=nb_classes,
                           random_state=random_state)
    # train and test split
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
        X, Y, S, test_size=test_split_size, stratify=Y,
        random_state=random_state)

    # Create a validation to be used to track progress
    X_train, X_val, Y_train, Y_val, S_train, S_val = train_test_split(
        X_train, Y_train, S_train, test_size=validation_split_size,
        shuffle=True, stratify=Y_train, random_state=random_state)

    return Y_train, X_train, S_train, \
           Y_test, X_test, S_test, \
           Y_val, X_val, S_val


def slice_songs(X, Y, S, length=911):
    """Slices the spectrogram into sub-spectrograms according to length"""

    # Create empty lists for train and test sets
    artist = []
    spectrogram = []
    song_name = []

    # Slice up songs using the length specified
    for i, song in enumerate(X):
        slices = int(song.shape[1] / length)
        for j in range(slices - 1):
            spectrogram.append(song[:, length * j:length * (j + 1)])
            artist.append(Y[i])
            song_name.append(S[i])

    return np.array(spectrogram), np.array(artist), np.array(song_name)


def create_dataset(genre_folder='../data/dataset/genres_original', save_folder='../data/np_data',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512):
    """This function creates the dataset given a folder
     with the correct structure (project root/data/genres_original/[genres]/*.wav)
    and saves it to a specified folder."""

    # get list of all artists
    os.makedirs(save_folder, exist_ok=True)
    # print(os.listdir(genre_folder),os.path.isdir('country'))
    # extract all genres' folders
    genres = [path for path in os.listdir(genre_folder)]

    samples = []
    labels = []
    max_shape = 0
    # iterate through all artists, albums, songs and find mel spectrogram
    for genre in tqdm(genres):
        print(genre)
        # e.g. ./data/generes_original/country
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
            if log_S.shape[1] != 94:
                continue
            # sample = np.expand_dims(log_S, axis=2)
            # sample = pad_along_axis(log_S, 1024, axis=1)
            samples.append(log_S)
            labels.append(genre)
            # target_folder = save_folder + '/' + genre
            # os.makedirs(target_folder, exist_ok=True)
            # save_name = song.split('.wav')[0] + '.png'
            # save_path = target_folder + '/' + save_name
            if max_shape < log_S.shape[1]:
                max_shape = log_S.shape[1]

            print(song_path, log_S.shape)
            # librosa.display.specshow(sample)
            # pylab.savefig(save_path, bbox_inches='tight', pad_inches=0)
            # pylab.close()

            # data = (genre, log_S, song)
            #
            # # Save each song
            # save_name = genre + '_' + song
            # with open(os.path.join(save_folder, save_name), 'wb') as fp:
            #     dill.dump(data, fp)
    samples = np.array(samples)
    labels = np.array(labels)

    np.save('../data/dataset/slice3s/validation/samples_val.npy', samples)
    np.save('../data/dataset/slice3s/validation/labels_val.npy', labels)
    print(max_shape)


def pad_along_axis(arr, target_length, axis):
    pad_size = target_length - arr.shape[axis]

    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)

    return np.pad(arr, pad_width=npad, mode='constant', constant_values=0)


def audio_clips(genre_folder='../data/dataset/genres_original',
                save_folder='../data/dataset/slice3s'):
    genres = [path for path in os.listdir(genre_folder)]
    addresses = []
    for genre in tqdm(genres):
        print(genre)
        # e.g. ./data/generes_original/country
        genre_path = os.path.join(genre_folder, genre)
        # extract all sounds from genre_path
        songs = os.listdir(genre_path)

        for song in tqdm(songs):
            song_path = os.path.join(genre_path, song)
            addresses.append([song_path, genre, song])

    addresses = np.array(addresses)

    train, test = train_test_split(addresses, shuffle=True, random_state=42, test_size=0.15)
    train, val = train_test_split(train, shuffle=True, random_state=42, test_size=0.15)

    for song_path, genre, song in train:
        os.makedirs(os.path.join(save_folder + '/train', genre), exist_ok=True)
        chunk_and_save(save_folder=save_folder + '/train', song_path=song_path, genre=genre, song=song)

    for song_path, genre, song in test:
        os.makedirs(os.path.join(save_folder + '/test', genre), exist_ok=True)
        chunk_and_save(save_folder=save_folder + '/test', song_path=song_path, genre=genre, song=song)

    for song_path, genre, song in val:
        os.makedirs(os.path.join(save_folder + '/validation', genre), exist_ok=True)
        chunk_and_save(save_folder=save_folder + '/validation', song_path=song_path, genre=genre, song=song)


def chunk_and_save(song_path, chunk_length_ms=3000, save_folder=None, genre=None, song=None):
    audio = AudioSegment.from_file(song_path, "wav")
    chunks = make_chunks(audio, chunk_length_ms)
    for i, chunk in enumerate(chunks):
        chunk_name = save_folder + '/' + genre + '/' + song.split('.wav')[0] + "_chunk{0}.wav".format(i)
        chunk.export(chunk_name, format="wav")


if __name__ == '__main__':
    create_dataset(genre_folder='../data/dataset/slice3s/validation', save_folder='../data/np_data/slice3s/validation')
    # audio_clips()
    # plt.imshow(img)
    # plt.savefig('tests.png', bbox_inches='tight', pad_inches=0)
    # plt.show()

    # print(f"Label: {label}")
