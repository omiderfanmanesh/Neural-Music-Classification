import os
from shutil import copyfile

import librosa
import librosa.display
import librosa.display
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.random import RandomState
from tqdm import tqdm

print(torch.version.__version__)
from sklearn.model_selection import train_test_split
from audiomentations import Compose, PolarityInversion

np.random.seed(42)
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import make_chunks
from utils.augmentation import freq_mask, time_mask, time_warp


# matplotlib.use('Agg') # No pictures displayed


def create_npy_dataset(genre_folder='../data/dataset/genres_original', save_folder='../data/np_data',
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

            if log_S.shape[1] < 94:
                continue
            # sample = np.expand_dims(log_S, axis=2)
            sample = pad_along_axis(log_S, 1024, axis=1)
            samples.append(sample)
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

    np.save(save_folder + '/samples_train_pitch_shift.npy', samples)
    np.save(save_folder + '/labels_train_pitch_shift.npy', labels)
    # print(max_shape)


def pad_along_axis(arr, target_length, axis):
    pad_size = target_length - arr.shape[axis]

    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)

    return np.pad(arr, pad_width=npad, mode='constant', constant_values=0)


def extract_address(src):
    genres = [path for path in os.listdir(src)]
    addresses = []
    for genre in tqdm(genres):
        print(genre)
        # e.g. ./data/generes_original/country
        genre_path = os.path.join(src, genre)
        # extract all sounds from genre_path
        songs = os.listdir(genre_path)

        for song in tqdm(songs):
            song_path = os.path.join(genre_path, song)
            addresses.append([song_path, genre, song])

    addresses = np.array(addresses)
    return addresses


def split_dataset(genre_folder='../data/dataset/genres_original',
                  save_folder='../data/dataset/slice3s'):
    addresses = extract_address(genre_folder)
    train, test = train_test_split(addresses, shuffle=True, random_state=42, test_size=0.15)
    train, val = train_test_split(train, shuffle=True, random_state=42, test_size=0.15)

    for src, genre, song in train:
        os.makedirs(os.path.join(save_folder + '/train', genre), exist_ok=True)
        dst = os.path.join(save_folder + '/train', genre, song)
        copyfile(src, dst)

    for src, genre, song in test:
        os.makedirs(os.path.join(save_folder + '/test', genre), exist_ok=True)
        dst = os.path.join(save_folder + '/test', genre, song)
        copyfile(src, dst)

    for src, genre, song in val:
        os.makedirs(os.path.join(save_folder + '/validation', genre), exist_ok=True)
        dst = os.path.join(save_folder + '/validation', genre, song)
        copyfile(src, dst)


def audio_clips(src, dst, chunk_length_ms=3000):
    dataset_address = extract_address(src)
    for song_path, genre, song in dataset_address:
        os.makedirs(os.path.join(dst, genre), exist_ok=True)
        chunk_and_save(song_path=song_path, save_folder=dst, chunk_length_ms=chunk_length_ms, genre=genre,
                       song=song)


def chunk_and_save(song_path, chunk_length_ms, save_folder=None, genre=None, song=None):
    audio = AudioSegment.from_file(song_path, "wav")
    chunks = make_chunks(audio, chunk_length_ms)
    for i, chunk in enumerate(chunks):
        chunk_name = save_folder + '/' + genre + '/' + song.split('.wav')[0] + "_chunk{0}.wav".format(i)
        chunk.export(chunk_name, format="wav")


def create_noisy_dataset(src, dst, sr=16000):
    dataset_address = extract_address(src)
    for song_path, genre, song in tqdm(dataset_address):
        os.makedirs(os.path.join(dst, genre), exist_ok=True)
        samples, sr = librosa.load(song_path, sr=sr)
        augment = Compose([
            # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.5, p=0.5),
            # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            # PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            PolarityInversion()
        ])

        # Augment/transform/perturb the audio data
        augmented_samples = augment(samples=samples, sample_rate=sr)
        sf.write(dst + '/' + genre + '/' + song.split('.wav')[0] + '_polarity_inv.wav', augmented_samples,
                 samplerate=sr)


def spec_augmentation(sample_address, label_address):
    samples = np.load(sample_address)
    labels = np.load(label_address)

    new_samples = []
    new_labels = []
    for s, l in tqdm(zip(samples, labels)):
        s = np.expand_dims(s, axis=0)

        new_samples.append(s)
        new_labels.append(l)

        s = torch.from_numpy(s)

        s_time_mask = time_mask(s.clone())
        new_samples.append(s_time_mask.numpy())
        new_labels.append(l)

        s_time_wrap = time_warp(s.clone())
        new_samples.append(s_time_wrap.numpy())
        new_labels.append(l)

        s_freq_mask = freq_mask(s.clone())
        new_samples.append(s_freq_mask.numpy())
        new_labels.append(l)

        combined = time_mask(freq_mask(time_warp(s.clone()), num_masks=2),
                             num_masks=2)
        new_samples.append(combined.numpy())
        new_labels.append(l)

    new_samples = np.array(new_samples)
    new_labels = np.array(new_labels)

    # samples = np.concatenate((samples,new_samples))
    # labels = np.concatenate((labels, new_labels))

    np.save('../data/dataset/slice3s/test/samples_test_aug.npy', new_samples)
    np.save('../data/dataset/slice3s/test/labels_test_aug.npy', new_labels)


def plot_spec(file_address, sr=16000, n_mels=128,
              n_fft=2048, hop_length=512):
    y, sr = librosa.load(file_address, sr=sr)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                       n_fft=n_fft,
                                       hop_length=hop_length)
    log_S = librosa.amplitude_to_db(S, ref=1.0)

    plt.figure()
    librosa.display.specshow(log_S)
    plt.colorbar()
    plt.show()

    sample = np.expand_dims(log_S, axis=2)
    plt.figure()
    plt.imshow(sample)
    plt.show()
    print(sample.shape, log_S.shape)


if __name__ == '__main__':
    # plot_spec(file_address='../data/dataset/slice3s/train_pitch_shift/classical/classical.00000_pitch_shift.wav')
    create_noisy_dataset(
        src='/home/omid/OMID/projects/python/mldl/NeuralMusicClassification/data/dataset/slice3s/train',
        dst='/home/omid/OMID/projects/python/mldl/NeuralMusicClassification/data/dataset/slice3s/train_polarity_inv')
    # audio_clips(src='/home/omid/OMID/projects/python/mldl/NeuralMusicClassification/data/dataset/slice3s/train',
    #             dst='/home/omid/OMID/projects/python/mldl/NeuralMusicClassification/data/dataset/slice3s/train3s')
    # split_dataset()
    # augmentation(
    #     sample_address='/home/omid/OMID/projects/python/mldl/NeuralMusicClassification/data/dataset/slice3s/test/samples_test.npy',
    #     label_address='/home/omid/OMID/projects/python/mldl/NeuralMusicClassification/data/dataset/slice3s/test/labels_test.npy')
    # create_npy_dataset(
    #     genre_folder='/home/omid/OMID/projects/python/mldl/NeuralMusicClassification/data/dataset/slice3s/train_noisy',
    #     save_folder='/home/omid/OMID/projects/python/mldl/NeuralMusicClassification/data/np_data/train_noisy')
    # audio_clips()
    # plt.imshow(img)
    # plt.savefig('tests.png', bbox_inches='tight', pad_inches=0)
    # plt.show()

# print(f"Label: {label}")
