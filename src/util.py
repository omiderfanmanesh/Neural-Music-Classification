import os

import librosa
import librosa.display
import pylab
from tqdm import tqdm


# matplotlib.use('Agg') # No pictures displayed


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
            print(log_S.shape)
            librosa.display.specshow(log_S)
            pylab.savefig(save_path, bbox_inches='tight', pad_inches=0)
            pylab.close()

# from MusicDataLoader.Loader import GTZANDataset
# from torch.utils.data import DataLoader
#
# if __name__ == '__main__':
#     #
#     # create_dataset()
#
#     #example of data loader
#     # dataset = GTZANDataset()
#     # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
#     #
#     # # Display image and label.
#     # train_features, train_labels = next(iter(dataloader))
#     # print(f"Feature batch shape: {train_features.size()}")
#     # print(f"Labels batch shape: {train_labels.size()}")
#     # img = train_features[0].squeeze()
#     # label = train_labels[0]
#     # plt.imshow(img)
#     # plt.savefig('tests.png', bbox_inches='tight', pad_inches=0)
#     # plt.show()
#     #
#     # print(f"Label: {label}")
