# in order to make this code work the there need to be
# the folder DatasetRawspectrograms_max
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
import swifter


def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(42)
    folders = sorted(os.listdir(r"Data\\genres_original"))
    folders_for_training = []
    for folder in folders:
        dir = get_png_from_wav(folder)
        if len(dir) == 2:
            folders_for_training.append([dir[0], dir[1]])
        elif len(dir) == 1:
            folders_for_training.append([dir[0]])
        else:
            print("We have a problem")
            return None

    path_to_model_to_try = r'CRNN.pth'  # you have to rename the model name cancelling the loss
    path_to_model = pathlib.Path(path_to_model_to_try)

    if path_to_model.is_file():
        print("Loading model")
        model = torch.load(path_to_model_to_try).to(device)

    else:
        print("Training model")
        model = Model().to(device)
        # path_to_model_to_train = r"CRNN_10_LOSS_0.779.pth"
        # model = torch.load(path_to_model_to_train).to(device)

        trainIters(model, 50)  # model, epochs,...

    model.eval()

    list_of_test_folders = os.listdir(r"Data\\DatasetRawspectrograms_max\\genres_png_validation")

    prediction_list = []

    pred_dict = {0: "blues", 1: "classical", 2: "country", 3: "disco",
                 4: "hiphop", 5: "jazz", 6: "metal", 7: "pop",
                 8: "reggae", 9: "rock"}

    for image_in_list in list_of_test_folders:
        current_spectrogram_path = pjoin(r"Data\\DatasetRawspectrograms_max\\genres_png_validation", f"{image_in_list}")

        spectrogram_to_predict = np.load(current_spectrogram_path)
        # spectrogram_to_predict = torch.round(spectrogram_to_predict/4)

        # eval_transforms = transforms.Compose([transforms.Resize((128,128),interpolation=Image.NEAREST),
        #                                       transforms.ToTensor(),
        #                                       # transforms.Normalize((0.1307,), (0.3081,)),
        #
        #                                       ])
        # x = eval_transforms(img)  # Preprocess image
        x = spectrogram_to_predict.unsqueeze(0).to(device)  # Add batch dimension
        x = torch.reshape(x, (x.shape[0], 1, 128, 128))

        model_hidden = model.initHidden()

        output = model(x, model_hidden)  # Forward pass
        pred = torch.argmax(output, 1)

        prediction_list.append([image_in_list[:-4], pred_dict[pred[0].tolist()]])

    return prediction_list


def get_png_from_wav(dir):
    current_dir = f"Data\\genres_original\\{dir}"
    list_of_wav = os.listdir(current_dir)
    dir_to_png = pjoin(f"Data\\DatasetRawspectrograms_max\\genres_png_training")
    dir_to_test = pjoin(f"Data\\DatasetRawspectrograms_max\\genres_png_test")
    dir_to_val = r"Data\\DatasetRawspectrograms_max\\genres_png_validation"

    j = 0
    for wav in list_of_wav:
        if (dir == "jazz") & (wav[-6:-4] == "54"):  ### file jazz.00054.wav might be broken!
            pass
        else:
            if j < 70:
                save_png_in_folders(current_dir, wav, dir_to_png, dir)
                j += 1
            elif j < 90:
                save_png_in_folders(current_dir, wav, dir_to_test, dir)
                j += 1
            else:
                save_png_in_folders(current_dir, wav, dir_to_val, dir)
                j += 1
    return dir_to_png, dir_to_test


def save_png_in_folders(current_dir, wav, second_directory, dir):
    if wav[-3:] == "wav":
        wav_file_path = pjoin(current_dir, wav)

        if second_directory != r"Data\\genres_png_validation":
            newpath = pjoin(second_directory, f"{dir}")
            if not os.path.exists(newpath):
                os.makedirs(newpath)
                save_wav_to_png(wav, newpath, wav_file_path)
            else:
                save_wav_to_png(wav, newpath, wav_file_path)
            return None

        elif second_directory == r"Data\\genres_png_validation":
            save_wav_to_png(wav, second_directory, wav_file_path)
            return None

    else:
        return None


def save_wav_to_png(wav, newpath, wav_file_path):
    png_file = wav[:-3]
    png_split = png_file.split(".")
    png_file = png_split[0] + "_" + png_split[1]
    # png_file_path = pjoin(newpath,
    #                       png_file)
    # samplerate, data = wavfile.read(wav_file_path)
    # n_fft = 2048
    # hop_length = 512
    mods_dict = {0: "normal", 1: "Compression", 2: "Shift", 3: "Stretch"}
    counter_mods = 0
    while counter_mods < 4:
        max_iteration = 28
        rate_stretching = 1
        if counter_mods == 3:
            rate_stretching = 2
            max_iteration /= rate_stretching
            max_iteration = np.int(max_iteration) - 1
        arguments = []
        first_cycle = True
        for sec in np.arange(0, max_iteration):
            png_file_path_with_sec = pjoin(newpath,
                                           f"Prova_Mod{mods_dict[counter_mods]}_Sec{sec}_{png_file}")
            if pathlib.Path(png_file_path_with_sec + ".npy").is_file():
                pass
            elif first_cycle:
                data, samplerate = librosa.load(wav_file_path)
                # spec = librosa.feature.melspectrogram(data, sr=samplerate, n_fft=2048, hop_length=512, n_mels=128)
                # spec = librosa.power_to_db(spec, ref=1.0)
                arguments.append([sec, data, samplerate, counter_mods, rate_stretching, png_file_path_with_sec])
                first_cycle = False
            else:
                arguments.append([sec, data, samplerate, counter_mods, rate_stretching, png_file_path_with_sec])
        if arguments:
            with multiprocessing.Pool(processes=4) as pool:
                pool.starmap(spectrogram_to_jpg, arguments)

        # for sec in np.arange(0, max_iteration):
        # spectrogram_to_jpg(sec, data, samplerate, counter_mods, rate_stretching, mods_dict, newpath, png_file, n_fft, hop_length)
        counter_mods += 1

        return None  # one tab back and it will do the mods


"""
This function receives the parameters needed to create a png picture for each wav file received
sec = starting second of the window
data = data of wav file to be sliced
samplerate = samplerate
counter_mods = {0: "normal", 1: "Compression", 2: "Shift", 3: "Stretch"}
rate_stretching = rate for time streactching (default 1)
png_file_path_with_sec = path to file
n_fft = 1024
hop_length = 512
"""


def spectrogram_to_jpg(sec, data, samplerate, counter_mods, rate_stretching, png_file_path_with_sec, ):
    y = data[
        sec * samplerate * rate_stretching:sec * samplerate * rate_stretching + 3 * samplerate * rate_stretching]

    if (counter_mods == 3) & (sec == 0):
        y = data[sec * samplerate:sec * samplerate * rate_stretching + 3 * samplerate * rate_stretching]

    if counter_mods == 2:
        y = librosa.effects.pitch_shift(y, samplerate, n_steps=4)
    if counter_mods == 3:
        y = librosa.effects.time_stretch(y, rate_stretching)
    spec = librosa.feature.melspectrogram(y, sr=samplerate, n_fft=2048, hop_length=512, n_mels=128)

    spec = librosa.power_to_db(spec, ref=np.max)

    if counter_mods == 1:
        threshold = np.average(spec)
        ratio = 4
        spec = pd.DataFrame(spec).applymap(
            lambda f: (threshold + (f - threshold) / (ratio)) if f > threshold else f).to_numpy()

    spec = spec[:, :128]
    if spec.shape == (128, 128):
        np.save(png_file_path_with_sec, spec)
        # res = librosa.feature.inverse.mel_to_audio(spec)
        # librosa.display.specshow(spec)
        # plt.show()
        # plt.savefig(png_file_path_with_sec, bbox_inches="tight")
        print(png_file_path_with_sec)

    # if sec == (max_iteration-1):
    #     wavfile.write("check.wav", samplerate, res)
    # plt.clf()


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 7, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.gru1 = nn.GRU(128, 32, num_layers=2, batch_first=True)
        self.out = nn.Linear(32, 10)

        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)

        self.mp1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.mp2 = nn.MaxPool2d((4, 2), stride=(4, 2))

        self.do_conv = nn.Dropout2d(p=0.1)  # 0.1
        self.do_gru = nn.Dropout(p=0.3)  # 0.3

        self.elu = nn.ELU()
        self.logSm = nn.LogSoftmax(dim=1)

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

        pass

    def initHidden(self):
        return torch.zeros((2, 16, 32)).to(self.device)

    def forward(self, x, hidden):
        x = self.bn0(x)

        x = self.elu(self.conv1(x))
        x = self.bn1(x)
        x = self.do_conv(x)

        x = self.mp1(x)

        x = self.elu(self.conv2(x))
        x = self.bn2(x)
        x = self.mp2(x)
        x = self.do_conv(x)

        x = self.elu(self.conv3(x))
        x = self.bn2(x)
        x = self.mp2(x)
        x = self.do_conv(x)

        x = self.elu(self.conv4(x))
        x = self.bn2(x)
        x = self.mp2(x)
        x = self.do_conv(x)

        x = x.permute(0, 3, 1, 2)
        resize_shape = list(x.shape)[2] * list(x.shape)[3]
        x = torch.reshape(x, (list(x.shape)[0], list(x.shape)[1], resize_shape))

        x, _ = self.gru1(x, hidden)
        x = self.do_gru(x)
        x = x[:, -1, :]
        x = self.out(x)
        x = self.logSm(x)
        return x


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def npy_loader(path):
    array_to_import = np.load(path)
    if array_to_import.shape == (128, 128):
        tensor = torch.from_numpy(array_to_import)
        return tensor
    else:
        return torch.zeros([128, 128])


def dynamic_range_compression(db):
    threshold = 0
    ratio = random.randint(2, 20)
    if db > threshold:
        return threshold + (db - threshold) / (ratio)
    else:
        return db


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainIters(model, epochs, print_every=1, plot_every=1, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    print(count_parameters(model))
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_transforms = transforms.Compose([transforms.Resize((128, 128), interpolation=Image.NEAREST),
                                           transforms.ToTensor(),
                                           # transforms.Normalize((0.1307,), (0.3081,)),
                                           # AddGaussianNoise(0., 1.)
                                           ])
    test_transforms = transforms.Compose([transforms.Resize((128, 128), interpolation=Image.NEAREST),
                                          transforms.ToTensor(),

                                          # transforms.Normalize((0.1307,), (0.3081,)),

                                          ])
    train_data = datasets.DatasetFolder(r"Data\\DatasetRawspectrograms_max\\genres_png_training",
                                        extensions="npy",
                                        loader=npy_loader)
    # transform=train_transforms)
    test_data = datasets.DatasetFolder(r"Data\\DatasetRawspectrograms_max\\genres_png_test",
                                       extensions="npy",
                                       loader=npy_loader)
    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    criterion = nn.NLLLoss()

    train_losses, test_losses = [], []
    running_loss = 0
    # size_to_check = torch.Size([128, 128])
    for current_epoch in range(1, epochs + 1):

        trn_corr = 0
        tot_iter = train_loader.sampler.num_samples
        for i, (X_train, y_train) in enumerate(train_loader):
            # if (i * 32 * 100) / tot_iter > 49.64:
            if X_train.shape[0] == batch_size:
                """
                # Check for wrong shape

                temporary_x = X_train
                for j, x_check in enumerate(X_train):
                    if (i==0) & (j==0):
                    # if (j == 0):
                        y_to_substitute = x_check
                    # shape_to_check = x_check.shape
                    boolean_check = torch.sum(x_check).numpy()
                    if not boolean_check:
                        indexes_to_keep = np.concatenate([np.arange(0, j), np.arange(j + 1, batch_size)])
                        list_of_tensors = [torch.squeeze(torch.index_select(temporary_x, 0, torch.LongTensor([index]))) for index in indexes_to_keep]
                        list_of_tensors.append(y_to_substitute)
                        temporary_x = torch.stack(list_of_tensors, 0)
                X_train = temporary_x

                """
                # """
                # Data augmentation through dynamic range compression
                # """
                # list_of_tensors_augmented = []
                # for j, x_to_augment in enumerate(X_train):
                #     if np.random.randn() > -1:
                #         # x_to_augment = pd.DataFrame(x_to_augment.numpy()).swifter.progress_bar(False).apply(lambda db_series: db_series.apply(lambda db: dynamic_range_compression(db))).to_numpy().astype("float")
                #         ratio = random.randint(2, 20)
                #         threshold = random.randint(0, 10)
                #         df_aug = pd.DataFrame(x_to_augment.numpy())
                #         mask1 = df_aug>threshold
                #         mask2 = df_aug<-threshold
                #         df_aug[mask1] = threshold + (df_aug[mask1] - threshold) / (ratio)
                #         df_aug[mask2] = -threshold + (df_aug[mask2] + threshold) / (ratio)
                #     list_of_tensors_augmented.append(torch.Tensor(x_to_augment))
                #
                # X_train = torch.stack(list_of_tensors_augmented, 0)

                # X_train = torch.round(X_train/4)

                progress = (i * batch_size * 100) / tot_iter
                print(f"\r{progress:.3f} %", end='')

                X_train, y_train = X_train.to(device), y_train.to(device)
                X_train = torch.reshape(X_train, (X_train.shape[0], 1, 128, 128))
                model_hidden = model.initHidden()
                optimizer.zero_grad()
                y_pred = model(X_train, model_hidden)
                loss = criterion(y_pred, y_train)
                predicted = torch.max(y_pred.data, 1)[1]
                batch_corr = (predicted == y_train).sum()
                trn_corr += batch_corr
                # optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

                print_loss_total += loss.item()
                plot_loss_total += loss.item()
                running_loss += loss.item()

        if current_epoch % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # inputs = torch.round(inputs/4)

                    inputs = torch.reshape(inputs, (inputs.shape[0], 1, 128, 128))

                    model_hidden = model.initHidden()
                    logps = model.forward(inputs, model_hidden)

                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))
            print()
            print(f"Epoch {current_epoch}/{epochs}.. "
                  f"Train loss: {running_loss / (i + 1) * print_every:.3f}.. "
                  f"Test loss: {test_loss / len(test_loader):.3f}.. "
                  f"Test accuracy: {accuracy / len(test_loader):.3f}")

            running_loss = 0
            model.train()

            print_loss_avg = print_loss_total / (i + 1) * print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, current_epoch / epochs),
                                         current_epoch, current_epoch / epochs * 100, print_loss_avg))

        if current_epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / (i + 1) * plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        torch.save(model, f'CRNN_LOSS_{print_loss_avg:.3f}.pth')

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



