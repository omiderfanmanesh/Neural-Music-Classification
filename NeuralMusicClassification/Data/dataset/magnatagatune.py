import os
from os.path import dirname, join as pjoin
import pathlib
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import random
from pydub import AudioSegment
import pandas as pd
import pydub
AudioSegment.ffmpeg = "C:\\Users\\Giovanni Calleris\\Desktop\Phyton\\PyCharm\\ffmpeg"

def get_spectrogram_from_wav():
    """It converts wav file to spectrogram using the save_spectrogram_in_folders function
    It creates the split of the songs in train/test/validation"""
    current_dir = f"Magnatagatune_dataset"
    df = pd.read_csv("annotations_final.csv", sep="\t")
    df["mp3_path"] = df["mp3_path"].apply(lambda n: n.split("/")[1])
    df = df.sort_values(by=["mp3_path"])
    df = df.drop(columns=["clip_id", "mp3_path"])
    list_of_album = os.listdir(current_dir)
    dict_of_songs_in_album = {}
    for album in list_of_album:
        dir_to_songs = current_dir + "\\" + album
        list_of_songs = os.listdir(dir_to_songs)
        for song in list_of_songs:
            dir_to_each_song = dir_to_songs + "\\" + song
            dict_of_songs_in_album[song] = dir_to_each_song

    dict_of_songs_in_album = sorted(dict_of_songs_in_album.items())
    list_of_songs_in_album = []
    for counter_song, el in enumerate(dict_of_songs_in_album):
        list_of_songs_in_album.append([el[1], df.iloc[counter_song].to_numpy()])
    random.seed(42)
    list_of_wav = random.sample(list_of_songs_in_album, len(list_of_songs_in_album))
    dir_to_training = "Magnatagatune_dataset_spectrogram/training"
    dir_to_test = "Magnatagatune_dataset_spectrogram/test"
    dir_to_val = "Magnatagatune_dataset_spectrogram/validation"
    j = 0
    for wav in list_of_wav:
        if j <= np.round(80*len(list_of_wav)/100):
            save_spectrogram_in_folders(current_dir, wav, dir_to_training)
            j += 1
        elif np.round(80*len(list_of_wav)/100) < j <= np.round(90*len(list_of_wav)/100):
            save_spectrogram_in_folders(current_dir, wav, dir_to_test)
            j += 1
        else:
            save_spectrogram_in_folders(current_dir, wav, dir_to_val)
            j += 1
        print(j*100/len(list_of_wav))
    return None


def save_spectrogram_in_folders(current_dir, wav, second_directory):
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
    if wav[0][-3:] == "mp3":
        # wav_file_path = pjoin(current_dir, wav)
        wav_file_path = wav[0]
        wav[0] = wav[0].split("\\")[-1][:-4]
        newpath = second_directory
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

    spectrogram_file = wav[0]
    counter_skipped = 0
    first = True
    max_iteration = 25
    try:
        for sec in np.arange(0, max_iteration, 3):
            spectrogram_file_path_with_sec = pjoin(newpath,
                                                   f"{wav[0]}_Sec{sec}")
            if pathlib.Path(spectrogram_file_path_with_sec + ".npy").is_file():
               pass
            else:
                if first:
                    dst = "TEMPORARY_WAV.wav"
                    sound = AudioSegment.from_mp3(wav_file_path)
                    sound.export(dst, format="wav")
                first = False

                spectrogram_to_npy(sec, wav_file_path, spectrogram_file_path_with_sec, dst, wav)
    except pydub.exceptions.CouldntDecodeError:
        counter_skipped += 1
        print(counter_skipped)
    return None


def spectrogram_to_npy(sec, wav_file_path, spectrogram_file_path_with_sec, dst, wav):
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
    spec = librosa.feature.melspectrogram(y, sr=samplerate, n_fft=2048, hop_length=512, n_mels=128)
    spec = librosa.power_to_db(spec, ref=1)
    spec = [spec, wav[1]]
    # plt.figure()
    # librosa.display.specshow(spec[0])
    # plt.colorbar()
    # plt.show()


    np.save(spectrogram_file_path_with_sec, spec)

    print(spectrogram_file_path_with_sec)


if __name__ == "__main__":
    #torch.manual_seed(16000)
    np.random.seed(42)
    #random.seed(16000)
    # folders = sorted(os.listdir(r"Magnatagatune_dataset"))
    # for folder in folders:
    get_spectrogram_from_wav()
