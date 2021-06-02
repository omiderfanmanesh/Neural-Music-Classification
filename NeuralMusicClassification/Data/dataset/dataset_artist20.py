import os
from os.path import dirname, join as pjoin
import pathlib
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import random
from pydub import AudioSegment
AudioSegment.ffmpeg = "C:\\Users\\Giovanni Calleris\\Desktop\Phyton\\PyCharm\\ffmpeg"

def get_spectrogram_from_wav(dir):
    """It converts wav file to spectrogram using the save_spectrogram_in_folders function
    It creates the split of the songs in train/test/validation"""
    current_dir = f"artist20\\clipsfull\\{dir}"
    list_of_album = os.listdir(current_dir)
    list_of_songs_in_album= []
    for album in list_of_album:
        dir_to_songs = current_dir + "\\" + album
        list_of_songs = os.listdir(dir_to_songs)
        for song in list_of_songs:
            dir_to_each_song = dir_to_songs + "\\" + song
            list_of_songs_in_album.append(dir_to_each_song)
    random.seed(42)
    list_of_wav = random.sample(list_of_songs_in_album, len(list_of_songs_in_album))
    dir_to_training = "artist20_npy/genres_spectrogram_training"
    dir_to_test = "artist20_npy/genres_spectrogram_test"
    dir_to_val = "artist20_npy/genres_spectrogram_validation"
    j = 0
    for wav in list_of_wav:
        if j <= np.round(80*len(list_of_wav)/100):
            save_spectrogram_in_folders(current_dir, wav, dir_to_training, dir)
            j += 1
        elif np.round(80*len(list_of_wav)/100) < j <= np.round(90*len(list_of_wav)/100):
            save_spectrogram_in_folders(current_dir, wav, dir_to_test, dir)
            j += 1
        else:
            save_spectrogram_in_folders(current_dir, wav, dir_to_val, dir)
            j += 1
    return None


def save_spectrogram_in_folders(current_dir, wav, second_directory, dir):
    """"

    Checks the files have the wav extension and
    use the save_wav_to_spectrogram function
    to convert wav to spectrogram

    Parameters:
        current_dir: the path to the current genre of wav files
        wav: a single wav contained in the current_dir
        second_directory: directory where the numpy array will be saved
        dir: the directory of the genre on which computing the split

    """
    if wav[-3:] == "mp3":
        # wav_file_path = pjoin(current_dir, wav)
        wav_file_path = wav
        wav = wav.split("\\")[-1]
        newpath = pjoin(second_directory, f"{dir}")
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        save_wav_to_spectrogram(wav, newpath, wav_file_path)
    return None


def save_wav_to_spectrogram(wav, newpath, wav_file_path):
    """"

    Creates a list of parameters which will be passed in parallel to the spectrogram_to_jpg function

    Parameters:
        wav: a single wav contained in the current_dir
        newpath: the path to the folder where the numpy array will be saved
        wav_file_path: the path to a single wav contained in the current_dir

    """

    spectrogram_file = wav[:-4]
    # spectrogram_split = spectrogram_file.split(".")
    # spectrogram_file = spectrogram_split[0] + "_" + spectrogram_split[1]
    first = True
    max_iteration = 28
    for sec in np.arange(0, max_iteration, 3):
        spectrogram_file_path_with_sec = pjoin(newpath,
                                               f"{spectrogram_file}_Sec{sec}")
        if pathlib.Path(spectrogram_file_path_with_sec + ".npy").is_file():
           pass
        else:
            if first:
                dst = "TEMPORARY_WAV.wav"
                sound = AudioSegment.from_mp3(wav_file_path)
                sound.export(dst, format="wav")
            first = False

            spectrogram_to_npy(sec, wav_file_path, spectrogram_file_path_with_sec, dst)
    return None


def spectrogram_to_npy(sec, wav_file_path, spectrogram_file_path_with_sec, dst):
    """
    It creates 10 non-overlapping audio clips of lenght of 3 seconds
    and it saves them as numpy array.

    Parameters:
        sec = the offset of the librosa.load function
        wav_file_path = the path to the wav file
        spectrogram_file_path_with_sec = the path to where the numpy array will be saved
    """

    # wav_file_path = pjoin(r"C:\\Users\\Giovanni Calleris\\Desktop\\Phyton\\PyCharm\\NeuralMusicClassification\\data\\dataset", wav_file_path)

    y, samplerate = librosa.load(dst, sr=16000, offset=sec, duration=3)
    #print(y)
    spec = librosa.feature.melspectrogram(y, sr=samplerate, n_fft=2048, hop_length=512, n_mels=128)
    spec = librosa.power_to_db(spec, ref=1)
    # rms = librosa.feature.rms(y)
    # chroma_stft = librosa.feature.chroma_stft(y, samplerate)
    # spectral_centroid = librosa.feature.spectral_centroid(y, samplerate)
    # spectral_bandwidth = librosa.feature.spectral_bandwidth(y, samplerate)
    # spectral_rolloff = librosa.feature.spectral_rolloff(y, samplerate)
    # zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    # harmonic = librosa.effects.harmonic(y)
    # tempo = librosa.beat.tempo(y, samplerate)
    # mfcc = librosa.feature.mfcc(y, samplerate)



    # features1 = np.vstack([rms.mean(), chroma_stft.mean(), spectral_centroid.mean(), spectral_bandwidth.mean(), spectral_rolloff.mean(),
    #                       zero_crossing_rate.mean(), harmonic.mean(), tempo, mfcc.mean()])
    # features2 = np.vstack(
    #     [rms.std(), chroma_stft.std(), spectral_centroid.std(), spectral_bandwidth.std(), spectral_rolloff.std(),
    #      zero_crossing_rate.std(), harmonic.std(), tempo, mfcc.std()])
    #
    # features = np.stack([features1, features2])
    plt.figure()
    librosa.display.specshow(spec)
    plt.colorbar()
    plt.show()


    np.save(spectrogram_file_path_with_sec, spec)

    print(spectrogram_file_path_with_sec)


if __name__ == "__main__":
    #torch.manual_seed(16000)
    np.random.seed(16000)
    #random.seed(16000)
    folders = sorted(os.listdir(r"artist20\\clipsfull"))
    for folder in folders:
        get_spectrogram_from_wav(folder)
