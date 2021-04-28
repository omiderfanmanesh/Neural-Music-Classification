# in order to make this code work the there need to be
# the folders: genres_png_training , genres_png_test , genres_png_validation
# inside the Data folder (which must be in the same folder of the the code NMC.py).
# The dataset is available at
# https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification

# it is difficult to replicate the pad=same parameter of the corresponing
# conv2d function of the keras library

# Epoch 20/50.. Train loss: 2.231.. Test loss: 2.246.. Test accuracy: 0.197
# at Epoch 50/50.. Train loss: 2.092.. Test loss: 2.183.. Test accuracy: 0.274

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

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,3)
        self.conv2 = nn.Conv2d(64,128,3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.gru1 = nn.GRU(2432, 128)
        self.gru2 = nn.GRU(128, 128)
        self.out = nn.Linear(128, 10)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.mp2 = nn.MaxPool2d((4, 2), stride=(4, 2))
        self.do_conv = nn.Dropout(p=0.1)
        self.do_gru = nn.Dropout(p=0.3)
        self.sm = nn.Softmax(dim=1)
        self.elu = nn.ELU()


        pass

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")

        # x = F.pad(x, (0, 0, 1, 2))
        x = self.elu(self.conv1(x))
        x = self.bn1(x)
        x = self.mp1(x)
        x = self.do_conv(x)

        # x = F.pad(x, (0, 0, 1, 2))
        x = self.elu(self.conv2(x))
        x = self.bn2(x)
        x = self.mp2(x)
        x = self.do_conv(x)

        # x = F.pad(x, (0, 0, 1, 2))
        x = self.elu(self.conv3(x))
        x = self.bn2(x)
        x = self.mp2(x)
        x = self.do_conv(x)

        # x = F.pad(x, (0, 0, 1, 2))
        x = self.elu(self.conv4(x))
        x = self.bn2(x)
        x = self.mp2(x)
        x = self.do_conv(x)

        x = x.permute(0, 2, 1, 3)
        resize_shape = list(x.shape)[2] * list(x.shape)[3]
        x = torch.reshape(x, (list(x.shape)[0], list(x.shape)[1], resize_shape))

        # x = x.view(-1, 128*5*5)
        h0 = torch.zeros((1, 1, 128)).to(device)
        x, hidden = self.gru1(x, h0)
        x = self.elu(x)
        x, hidden = self.gru2(x, hidden)

        x = self.elu(x)
        x, hidden = self.gru2(x, hidden)
        x = self.do_gru(x)

        x = self.elu(x[:, -1, :])
        x = self.do_gru(x)


        x = self.out(x)
        x = self.sm(x)

        return x


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def trainIters(model, epochs, print_every=1, plot_every=1, learning_rate=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,)),
                                           AddGaussianNoise(0., 1.)
                                           ])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,)),

                                          ])
    train_data = datasets.ImageFolder(r"Data\\genres_png_training",
                                      transform=train_transforms)
    test_data = datasets.ImageFolder(r"Data\\genres_png_test",
                                     transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    train_losses, test_losses = [], []
    running_loss = 0

    for current_epoch in range(1, epochs + 1):

        trn_corr = 0

        for i, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.to(device), y_train.to(device)

            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr
            optimizer.zero_grad()
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
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))
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

def get_png_from_wav(dir):
    current_dir = f"Data\\genres_original\\{dir}"
    list_of_wav = os.listdir(current_dir)
    dir_to_png = pjoin(f"Data\\genres_png_training")
    dir_to_test = pjoin(f"Data\\genres_png_test")
    dir_to_val = r"Data\\genres_png_validation"

    j=0
    for wav in list_of_wav:
        if (dir == "jazz") & (wav[-6:-4] == "54"): ### file jazz.00054.wav might be broken!
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
    if wav[-3:]=="wav":
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



def save_wav_to_png(wav,newpath, wav_file_path):
    png_file = wav[:-3] + "png"
    png_file_path = pjoin(newpath,
                          png_file)
    if pathlib.Path(png_file_path).is_file():
        pass
    else:
        samplerate, data = wavfile.read(wav_file_path)
        frequencies, times, spectrogram = signal.spectrogram(data, samplerate, nperseg=1024)

        # make black background
        # plt.rcParams['axes.facecolor'] = 'black'

        plt.pcolormesh(times, frequencies, 10*np.log10(spectrogram), shading='gouraud')


        plt.set_cmap('Spectral')
        plt.yscale('log')
        plt.xscale('linear')
        # fix axis
        plt.xlim(0, 30)
        plt.ylim(bottom=1, top=50000)

        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')

        # # remove axis (it also removes the black background)
        plt.axis('off')
        # plt.show()
        plt.savefig(png_file_path, bbox_inches='tight', pad_inches=0, dpi=255)
        plt.clf()

    return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(0)
    folders = os.listdir(r"Data\\genres_original")
    folders_for_training=[]
    for folder in folders:
        dir = get_png_from_wav(folder)
        if len(dir)==2:
            folders_for_training.append([dir[0], dir[1]])
        elif len(dir)==1:
            folders_for_training.append([dir[0]])
        else:
            print("We have a problem")
            return None
    path_to_model_to_try = r'CRNN.pth' # you have to rename the model name cancelling the loss
    path_to_model = pathlib.Path(path_to_model_to_try)


    if path_to_model.is_file():
        model = torch.load(r'CRNN.pth').to(device)

    else:
        model = Model().to(device)
        trainIters(model, 50) # model, epochs,...

    model.eval()

    list_of_test_folders = os.listdir(r"Data\\genres_png_validation")

    prediction_list = []

    pred_dict ={0: "blues", 1: "classical", 2: "country", 3: "disco",
                4: "hiphop", 5: "jazz", 6: "metal", 7: "pop",
                8: "reggae", 9: "rock"}
    
    for image_in_list in list_of_test_folders:
        current_image_path = pjoin(r"Data\\genres_png_validation", f"{image_in_list}")

        img = Image.open(current_image_path).convert('RGB')
        eval_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,)),

                                                ])
        x = eval_transforms(img)  # Preprocess image
        x = x.unsqueeze(0).to(device) # Add batch dimension

        output = model(x)  # Forward pass
        pred = torch.argmax(output, 1)

        prediction_list.append([image_in_list[:-4], pred_dict[pred[0].tolist()]])

    return prediction_list

result = main()
with open(r'result_CRNN.csv', "w", newline="") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerows(result)