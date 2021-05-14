# in order to make this code work the there need to be
# the folder Just_trying
# inside the Data folder (which must be in the same folder of the the code NMC.py).
# The dataset is available at
# https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification


# In order to laod the model
# you have to rename the model name cancelling the loss
# around line 330

import os
from os.path import dirname, join as pjoin
from scipy.io import wavfile
from scipy import signal
import pathlib
import csv
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import numpy as np
import importlib
import pandas as pd
import librosa
import librosa.display
import multiprocessing
import random
import argparse
from collections import defaultdict
import swifter
from sklearn.metrics import classification_report



def main():

    """"

    First, creates the dataset of numpy array from wav files
        using the get_spectrogram_from_wav function
    Second, if no model is loaded it trains a CRNN on the numpy array
    Last, checks the performance on the validation set

    """
    torch.manual_seed(16000)
    np.random.seed(16000)
    random.seed(16000)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    folders = sorted(os.listdir(r"Data\\genres_original"))
    for folder in folders:
        get_spectrogram_from_wav(folder)


    path_to_model_to_try = r"WRITE HERE THE MODEL NAME"  # you have to rename the model name
    path_to_model = pathlib.Path(path_to_model_to_try)

    if path_to_model.is_file():
        print("Loading model")
        model = torch.load(path_to_model_to_try).to(device)

    else:
        print("Training model")
        model = Model().to(device)
        # path_to_model_to_train = r"WRITE HERE THE MODEL NAME"
        # model = torch.load(path_to_model_to_train).to(device)
        trainIters(model, 100)  # model, epochs,...

    model.eval()

    list_of_test_folders = os.listdir(r"Data\\Just_trying\\genres_spectrogram_validation")

    prediction_list = []

    pred_dict = {0: "blues", 1: "classical", 2: "country", 3: "disco",
                 4: "hiphop", 5: "jazz", 6: "metal", 7: "pop",
                 8: "reggae", 9: "rock"}

    inv_pred_dict = {"blues": 0, "classical":1, "country": 2, "disco": 3,
                 "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7,
                 "reggae": 8, "rock": 9}
    target_names = list(inv_pred_dict.keys())
    tot_validation = 0
    tot_validation_30_sec = 0
    tot_correct = 0
    tot_correct_30_sec = 0
    dictionary_of_results={}
    for image_in_folder in list_of_test_folders:
        current_spectrogram_path_folder = pjoin(r"Data\\Just_trying\\genres_spectrogram_validation", f"{image_in_folder}")

        for image_in_list in os.listdir(current_spectrogram_path_folder):
            current_spectrogram_path = pjoin(current_spectrogram_path_folder, f"{image_in_list}")
            tot_validation += 1

            spectrogram_to_predict = npy_loader_test(current_spectrogram_path).to(device)
            spectrogram_to_predict = torch.reshape(spectrogram_to_predict, (1, 2, 128, 94))

            model_hidden = model.initHidden(spectrogram_to_predict)

            output = model(spectrogram_to_predict, model_hidden)  # Forward pass
            pred = torch.argmax(output, 1)
            genre = image_in_list.split("Sec")[-1].split("_")
            groud_truth= genre[1]
            dict_key = f"{genre[1]}_{genre[2]}"
            if dict_key not in dictionary_of_results:
                dictionary_of_results[dict_key] = [0,0,0,0,0,0,0,0,0,0]
            dictionary_of_results[dict_key][pred[0].tolist()] += 1

            if groud_truth == pred_dict[pred[0].tolist()]:
                tot_correct +=1

            prediction_list.append([image_in_list[:-4], pred_dict[pred[0].tolist()]])

    y_true = []
    y_pred = []
    for key in dictionary_of_results:
        tot_validation_30_sec += 1
        predicted = pred_dict[np.argmax(np.array(dictionary_of_results[key]))]
        y_pred.append(inv_pred_dict[predicted])
        y_true_single = key.split("_")[0]
        y_true.append(inv_pred_dict[y_true_single])
        if predicted == y_true_single:
            tot_correct_30_sec += 1
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(tot_correct/tot_validation)
    print(tot_correct_30_sec/tot_validation_30_sec)
    return prediction_list


def get_spectrogram_from_wav(dir):

    """"

    Converts wav file to spectrogram using the save_spectrogram_in_folders function.

    Creates the split of songs in train (81%), test (10%), validation (9%).

    Parameters:
        dir: the directory of the genre on which computing the split

    """

    current_dir = f"Data\\genres_original\\{dir}"
    list_of_wav = os.listdir(current_dir)
    dir_to_spectrogram = pjoin(f"Data\\Just_trying\\genres_spectrogram_training")
    dir_to_test = pjoin(f"Data\\Just_trying\\genres_spectrogram_test")
    dir_to_val = r"Data\\Just_trying\\genres_spectrogram_validation"

    j = 0
    for wav in list_of_wav:
        if (dir == "jazz") & (wav[-6:-4] == "54"):  ### file jazz.00054.wav might be broken!
            pass
        else:
            if j <= 80:
                save_spectrogram_in_folders(current_dir, wav, dir_to_spectrogram, dir)
                j += 1
            elif 80 < j <= 90:
                save_spectrogram_in_folders(current_dir, wav, dir_to_test, dir)
                j += 1
            elif 90<j<=100:
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

    if wav[-3:] == "wav":
        wav_file_path = pjoin(current_dir, wav)

        if second_directory != r"Data\\genres_spectrogram_validation":
            newpath = pjoin(second_directory, f"{dir}")
            if not os.path.exists(newpath):
                os.makedirs(newpath)
                save_wav_to_spectrogram(wav, newpath, wav_file_path)
            else:
                save_wav_to_spectrogram(wav, newpath, wav_file_path)
            return None

        elif second_directory == r"Data\\genres_spectrogram_validation":
            save_wav_to_spectrogram(wav, second_directory, wav_file_path)
            return None
    else:
        return None


def save_wav_to_spectrogram(wav, newpath, wav_file_path):

    """"

    Creates a list of parameters which will be passed in parallel to the spectrogram_to_jpg function

    Parameters:
        wav: a single wav contained in the current_dir
        newpath: the path to the folder where the numpy array will be saved
        wav_file_path: the path to a single wav contained in the current_dir

    """

    spectrogram_file = wav[:-3]
    spectrogram_split = spectrogram_file.split(".")
    spectrogram_file = spectrogram_split[0] + "_" + spectrogram_split[1]

    mods_dict = {0: "normal",}
    counter_mods = 0
    max_iteration = 28
    arguments = []
    first_cycle = True
    for sec in np.arange(0, max_iteration, 3):
        spectrogram_file_path_with_sec = pjoin(newpath,
                                       f"Prova_Mod{mods_dict[counter_mods]}_Sec{sec}_{spectrogram_file}")
        if pathlib.Path(spectrogram_file_path_with_sec + ".npy").is_file():
            pass
        elif first_cycle:

            arguments.append([sec, wav_file_path, spectrogram_file_path_with_sec])
            first_cycle = False
        else:
            arguments.append([sec, wav_file_path, spectrogram_file_path_with_sec])
    if arguments:
        with multiprocessing.Pool(processes=4) as pool:
            pool.starmap(spectrogram_to_jpg, arguments)
    return None





def spectrogram_to_jpg(sec, wav_file_path, spectrogram_file_path_with_sec, ):

    """
    This function works in parallel.
    It creates 10 anon-overlapping audio clips of lenght of 3 seconds
    and saves them as numpy array.

    Parameters:
        sec = the offset of the librosa.load function
        wav_file_path = the path to the wav file
        spectrogram_file_path_with_sec = the path to where the numpy array will be saved
    """

    y, samplerate = librosa.load(wav_file_path, sr=16000, offset=sec, duration=3)

    spec0 = librosa.feature.melspectrogram(y, sr=samplerate, n_fft=2048, hop_length=512, n_mels=128)
    spec0 = librosa.power_to_db(spec0, ref=1)
    C = librosa.cqt(y, sr=samplerate, n_bins=128, bins_per_octave=19)
    spec1 = librosa.amplitude_to_db(C**2)
    spec = np.stack([spec0, spec1, C])

    np.save(spectrogram_file_path_with_sec, spec)

    print(spectrogram_file_path_with_sec)


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.gru1 = nn.GRU(128, 32, num_layers=2, batch_first=True)
        self.out = nn.Linear(32, 10)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.mp2 = nn.MaxPool2d((4, 2), stride=(4, 2))
        self.mp3 = nn.MaxPool2d((4, 2), stride=(4, 2))
        self.mp4 = nn.MaxPool2d((4, 2), stride=(4, 2))
        self.do_conv = nn.Dropout(p=0.1)  # 0.1
        self.do_gru = nn.Dropout(p=0.3)  # 0.3
        self.logSm = nn.LogSoftmax(dim=1)
        self.elu = nn.ELU()

        # self.tanh = nn.Tanh()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

        pass

    def initHidden(self, x):
        if (self.training == False) & (x.shape == (1,2,128,94)):
            return torch.zeros((2, 1, 32)).to(self.device)
        else:
            return torch.zeros((2, 32, 32)).to(self.device)



    def forward(self, x, hidden):

        x = self.bn0(x)
        x = self.elu(self.conv1(x))
        x = self.bn1(x)
        x = self.mp1(x)
        x = self.do_conv(x)

        x = self.elu(self.conv2(x))
        x = self.bn2(x)
        x = self.mp2(x)
        x = self.do_conv(x)

        x = self.elu(self.conv3(x))
        x = self.bn2(x)
        x = self.mp3(x)
        x = self.do_conv(x)

        x = self.elu(self.conv4(x))
        x = self.bn2(x)
        x = self.mp4(x)
        x = self.do_conv(x)

        x = x.permute(0, 3, 1, 2)
        resize_shape = list(x.shape)[2] * list(x.shape)[3]
        x = torch.reshape(x, (list(x.shape)[0], list(x.shape)[1], resize_shape))

        x, _ = self.gru1(x, hidden)
        x = self.do_gru(x)
        x = x[:, -1, :]
        x = self.do_gru(x)
        x = self.out(x)
        return x

        pass





def npy_loader(path):

    """"

    Loads the numpy array to the training set

    """


    array_to_import = np.load(path)[0, :, :]

    if array_to_import.shape == (128, 94):

        # array_to_import = np.stack([np.real(array_to_import), np.imag(array_to_import)], axis=0)
        array_to_import = np.real(array_to_import)
        array_to_import = np.reshape(array_to_import, (1, 128, 94))
        # resize_shape = list(array_to_import.shape)[0] * list(array_to_import.shape)[1]
        # array_to_import = np.reshape(array_to_import,
        #                        [resize_shape, list(array_to_import.shape)[2], list(array_to_import.shape)[3]])

        tensor = torch.from_numpy(array_to_import)

        return tensor
    else:
        return torch.zeros([1, 128, 94])

def npy_loader_test(path):

    """"

    Loads the numpy array to the training set

    """

    array_to_import = np.load(path)[0,:,:]

    # array_to_import = np.stack([np.real(array_to_import), np.imag(array_to_import)], axis=0)
    array_to_import = np.real(array_to_import)
    array_to_import = np.reshape(array_to_import, (1, 128, 94))
    if array_to_import.shape == (1, 128, 94):
        tensor = torch.from_numpy(array_to_import)
        return tensor
    else:
        return torch.zeros([1, 128, 94])


def dynamic_range_compression(db):
    threshold = 0
    ratio = random.randint(2, 20)
    if db > threshold:
        return threshold + (db - threshold) / (ratio)
    else:
        return db


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def augmentation(df_aug, counter_imag):
    if np.random.randn() > 0:
        ratio = random.randint(2, 20)
        threshold = 1
        mask1 = df_aug > threshold
        mask2 = df_aug < -threshold
        df_aug[mask1] = threshold + (df_aug[mask1] - threshold) / (ratio)
        df_aug[mask2] = -threshold + (df_aug[mask2] + threshold) / (ratio)

    return torch.from_numpy(df_aug.to_numpy()), counter_imag


def trainIters(model, epochs, print_every=1, plot_every=1, learning_rate=0.0001): # 0.00008 # 0.001

    """"

    Trains the model on three different tasks:

        1) Recongnise the  genre of the clip
        2) Recognise the frequency range
        3) Recognise if the first and the second half of the clip have been swapped along the time axis

    """

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    print(count_parameters(model))
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # 001 # 00008 # , weight_decay=0.00008

    train_data = datasets.DatasetFolder(r"Data\\Just_trying\\genres_spectrogram_training",
                                        extensions="npy",
                                        loader=npy_loader)
    # transform=train_transforms)
    test_data = datasets.DatasetFolder(r"Data\\Just_trying\\genres_spectrogram_test",
                                       extensions="npy",
                                       loader=npy_loader_test)
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True) #, num_workers=4
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,)

    criterion = nn.CrossEntropyLoss()

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
    parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
    args = vars(parser.parse_args())

    if args['lr_scheduler']:
        print('INFO: Initializing learning rate scheduler')
        lr_scheduler = LRScheduler(optimizer)
        # change the accuracy, loss plot names and model name
        loss_plot_name = 'lrs_loss'
        acc_plot_name = 'lrs_accuracy'
        # model_name = 'lrs_model'
    if args['early_stopping']:
        print('INFO: Initializing early stopping')
        early_stopping = EarlyStopping()
        # change the accuracy, loss plot names and model name
        loss_plot_name = 'es_loss'
        acc_plot_name = 'es_accuracy'
        # model_name = 'es_model'


    train_losses, test_losses = [], []
    running_loss = 0

    # size_to_check = torch.Size([128, 128])
    for current_epoch in range(1, epochs + 1):

        trn_corr = 0
        tot_iter = train_loader.sampler.num_samples
        for i, (X_train, y_train) in enumerate(train_loader):
            if X_train.shape[0] == batch_size:

                progress = (i * batch_size * 100) / tot_iter
                print(f"\r{progress:.3f} %", end='')

                X_train, y_train = X_train.to(device), y_train.to(device)

                model_hidden = model.initHidden(X_train)
                optimizer.zero_grad()
                y_pred = model(X_train, model_hidden)
                loss = criterion(y_pred, y_train)

                predicted = torch.max(y_pred.data, 1)[1]
                batch_corr = (predicted == y_train).sum()
                trn_corr += batch_corr
                running_loss += loss


                loss.backward()
                optimizer.step()

                print_loss_total += loss.item()
                plot_loss_total += loss.item()
                if args['lr_scheduler']:
                    lr_scheduler(loss)
                if args['early_stopping']:
                    early_stopping(loss)
                    if early_stopping.early_stop:
                        break
        print(f"Epoch: {current_epoch}/{epochs}")

        if current_epoch % print_every == 0:
            test_loss = 0

            accuracy = 0
            accuracy2 = 0
            model.eval()
            with torch.no_grad():
                for index_test, (inputs, labels) in enumerate(test_loader):

                    if inputs.shape[0] == batch_size:

                        inputs, labels = inputs.to(device), labels.to(device)

                        model_hidden = model.initHidden(inputs)
                        logps = model.forward(inputs, model_hidden)

                        batch_loss = criterion(logps, labels)


                        test_loss += batch_loss


                        ps = torch.exp(logps)
                        sm_of_output = nn.functional.softmax(logps, dim=1)
                        accuracy2 += (labels == torch.argmax(sm_of_output, dim=1)).sum()

                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))
            print()
            print(f"Epoch {current_epoch}/{epochs}.. "
                  f"Train loss: {running_loss / ((i + 1)*print_every):.3f}.. "
                  f"Test loss: {test_loss/(index_test+1):.3f}.. "
                  # f"Test loss2: {test_loss2/(index_test+1):.3f}.. "
                  # f"Test loss3: {test_loss3/(index_test+1):.3f}.. "
                  f"Train accuracy: {trn_corr / ((i + 1)*batch_size):.3f}.. "
                  f"Test accuracy: {accuracy2 / ((index_test+1)*batch_size):.3f}"
                  )

            running_loss = 0


            model.train()

            print_loss_avg = print_loss_total / ((i + 1) * print_every)
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, current_epoch / epochs),
                                         current_epoch, current_epoch / epochs * 100, print_loss_avg))

            torch.save(model, f'CRNN_LOSS_{print_loss_avg:.3f}.pth')

        if current_epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / (i + 1) * plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


    showPlot(plot_losses)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


if __name__ == '__main__':
    result = main()
    with open(r'result_CRNN.csv', "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows(result)



