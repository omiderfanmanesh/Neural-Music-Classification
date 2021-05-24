import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn import svm
from scipy import stats

label_encoder = LabelEncoder()
train_boolean = False
deep_learning = False
do_svm = True

def accuracy_for_each_class(target_test, preds):
    table = pd.DataFrame({'accuracy': 0, 'class': label_encoder.classes_})
    class_correct = [0 for i in range(10)]
    class_total = [0 for i in range(10)]
    names = label_encoder.classes_

    c = np.array(np.array(preds) == np.array(target_test))
    for i in range(len(preds)):
        label = int(preds[i])
        class_correct[label] += c[i]
        class_total[label] += 1

    for i in range(10):
        table.loc[i, 'class'] = names[i]
        if class_total[i] != 0:
            table.loc[i, 'accuracy'] = 100 * class_correct[i] / class_total[i]
        else:
            table.loc[i, 'accuracy'] = -1

    table.index = table['class']

    return table

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

class MusicClassification(nn.Module):
    def __init__(self):
        super(MusicClassification, self).__init__()
        device = "cuda"

        self.activation = nn.ReLU()
        self.lin1 = nn.Linear(57, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 256)
        self.lin4 = nn.Linear(256, 128)
        self.lin5 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.4)
        self.norm1 = nn.BatchNorm1d(1024)
        self.norm2 = nn.BatchNorm1d(512)
        self.norm3 = nn.BatchNorm1d(256)
        self.norm4 = nn.BatchNorm1d(128)
        self.norm5 = nn.BatchNorm1d(64)
        self.dense = nn.Linear(64, 10)

        # torch.nn.init.xavier_uniform_(self.lin1.weight)
        # torch.nn.init.xavier_uniform_(self.lin2.weight)
        # torch.nn.init.xavier_uniform_(self.lin3.weight)
        # torch.nn.init.xavier_uniform_(self.lin4.weight)
        # torch.nn.init.xavier_uniform_(self.lin5.weight)


    def forward(self, x):

        x = self.norm1(self.dropout(self.activation(self.lin1(x))))
        x = self.norm2(self.dropout(self.activation(self.lin2(x))))
        x = self.norm3(self.dropout(self.activation(self.lin3(x))))
        x = self.norm4(self.dropout(self.activation(self.lin4(x))))
        x = self.norm5(self.dropout(self.activation(self.lin5(x))))
        x = self.dense(x)
        return x


SEED=42
random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

data = pd.read_csv("features_3_sec.csv")

dict_to_label = {"blues": 0, "classical":1, "country": 2, "disco": 3,
                 "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7,
                 "reggae": 8, "rock": 9}

all_labels = data['label'].apply(lambda name: dict_to_label[name])
# labels = data['label']
label_encoder.fit(all_labels)
data = data.drop(columns=["filename", "label", "length"])
columns = data.columns
"""
features, features_validation, target, target_validation = train_test_split(data, all_labels, test_size=0.25, random_state = SEED, shuffle = True)
"""

shape_0 = data.shape[0]
list_for_split = list(np.arange(0, int(shape_0/10)))
split = random.sample(list_for_split, int(len(list_for_split)))
data_to_stack = []
label_to_stack = []
for index in split:
    data_to_stack.append(data[index*10:index*10+10])
    label_to_stack.append(all_labels[index*10:index*10+10])

data = np.concatenate(data_to_stack)
all_labels = np.concatenate(label_to_stack)
data = pd.DataFrame(data, columns=columns)
all_labels = pd.DataFrame(all_labels, columns=["label"])


features, features_validation, features_test = data.iloc[:int(np.around(shape_0*0.81,-1))], data.iloc[int(np.around(shape_0*0.81,-1)):int(np.around(shape_0*0.90,-1))].values, data.iloc[int(np.around(shape_0*0.90,-1)):].values
target, target_validation, target_test = all_labels.iloc[:int(np.around(shape_0*0.81,-1))], all_labels.iloc[int(np.around(shape_0*0.81,-1)):int(np.around(shape_0*0.90,-1)),0].to_numpy(), all_labels.iloc[int(np.around(shape_0*0.90,-1)):,0].to_numpy()

# features = pd.DataFrame(features)
# target = pd.DataFrame(target)
features = pd.concat([features, target], axis=1)
features = features.sample(frac=1, axis=0)
target = features["label"].to_numpy()
features = features.drop(columns=["label"]).to_numpy()

scaler = StandardScaler()
features = pd.DataFrame(scaler.fit_transform(features), columns=columns)
features_validation = pd.DataFrame(scaler.transform(features_validation), columns=columns)
features_test = pd.DataFrame(scaler.transform(features_test), columns=columns)
# features, features_validation, target, target_validation = features.to_numpy(), features_validation.to_numpy(), target.to_numpy(), target_validation.to_numpy()
features, features_validation, features_test = features.to_numpy(), features_validation.to_numpy(), features_test.to_numpy()

target_names = list(dict_to_label.keys())
clf = svm.SVC(kernel="rbf", C=10, gamma=0.08)
print("HELLO")
if do_svm:
    clf.fit(features, target)
    print("HELLO")

    pred_svm=[]
    i=0
    while i <10:
        pred_svm.append(clf.predict(features_test[i::10]))
        i += 1
        print("HELLO")
    pred_svm, _ = stats.mode(np.stack(pred_svm, axis=0), axis=0)
    pred_svm = pred_svm[0,:]
    target_test = target_test[::10]
    print(classification_report(target_test, pred_svm, target_names=target_names))

if deep_learning:
    model = MusicClassification().to("cuda")
    learning_rate = 0.001
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # 001 # 00008

    acc_to_save = 0
    running_loss = 0
    print_every = 1
    plot_every= 1
    criterion = nn.CrossEntropyLoss()
    epochs = 100
    device = "cuda"


    if train_boolean:
        for current_epoch in range(1, epochs + 1):
            trn_corr = 0
            batch_size = 128

            tot_iter = int(np.around(features.shape[0]/batch_size))
            # tot_iter = train_loader.sampler.num_samples
            for i in np.arange(0,tot_iter):
                X_train = torch.tensor(features[i*batch_size:i*batch_size+batch_size]).type(torch.float)

                # if (i * 32 * 100) / tot_iter > 49.64:
                if X_train.shape[0] == batch_size:
                    progress = (i * 100) / tot_iter
                    print(f"\r{progress:.3f} %", end='')
                    y_train = torch.tensor(target[i*batch_size:i*batch_size+batch_size])

                    X_train, y_train = X_train.to(device), y_train.to(device)

                    optimizer.zero_grad()
                    y_pred = model(X_train)
                    loss = criterion(y_pred, y_train)


                    predicted = torch.max(y_pred.data, 1)[1]
                    batch_corr = (predicted == y_train).sum()
                    trn_corr += batch_corr
                    running_loss += loss


                    # loss2 /= 10
                    # loss3 /= 10

                    # loss3.backward(retain_graph=True)
                    loss.backward()
                    optimizer.step()

                    print_loss_total += loss.item()
                    plot_loss_total += loss.item()

            print(f"Epoch: {current_epoch}/{epochs}")

            if current_epoch % print_every == 0:
                validation_loss = 0

                accuracy = 0
                accuracy2 = 0
                accuracy1 = 0
                model.eval()
                # batch_size_2 = features_validation.shape[0]
                batch_size_2 = 10
                with torch.no_grad():
                    tot_iter_validation = int(np.around(features_validation.shape[0] / batch_size_2))

                    for j in np.arange(0, tot_iter_validation):
                        inputs = torch.tensor(features_validation[j*batch_size_2:j*batch_size_2+batch_size_2]).type(torch.float)
                        labels = torch.tensor(target_validation[j*batch_size_2:j*batch_size_2+batch_size_2])

                        if inputs.shape[0] == batch_size_2:
                            inputs, labels = inputs.to(device), labels.to(device)

                            predicted_validation = model.forward(inputs)

                            batch_loss = criterion(predicted_validation, labels)


                            validation_loss += batch_loss


                            sm_of_output = nn.functional.softmax(predicted_validation, dim=1)
                            # amax = torch.argmax(sm_of_output, dim=1)
                            # amax = torch.argmax(sm_of_output, dim=1)
                            # equals = torch.eq(labels, amax)
                            # accuracy2 += equals.sum()
                            # accuracy1 = accuracy_for_each_class(labels.cpu(), amax.cpu())

                            amax, _ = stats.mode(torch.argmax(sm_of_output, dim=1).cpu().numpy())
                            amax = torch.tensor(amax).repeat(batch_size_2).cuda()
                            equals = torch.eq(labels, amax)
                            accuracy2 += equals.sum()
                            accuracy1 = accuracy_for_each_class(labels.cpu(), amax.cpu())
                accuracy2 = accuracy2 / ((j + 1) * batch_size_2)
                # train_losses.append(running_loss / len(train_loader))
                # validation_losses.append(validation_loss / len(validation_loader))
                # print(accuracy1)
                print(f"Epoch {current_epoch}/{epochs}.. "
                      f"Train loss: {running_loss / ((i + 1) * print_every):.3f}.. "
                      
                      f"validation loss: {validation_loss / (j + 1):.3f}.. "
                     
                      f"Train accuracy: {trn_corr / ((i + 1) * batch_size):.3f}.. "
                      # f"validation accuracy_GTZAN: {accuracy1 / ((j + 1) * batch_size_2):.3f}"
                      f"validation accuracy: {accuracy2:.3f}"
                      )

                running_loss = 0
                running_loss2 = 0
                running_loss3 = 0

                model.train()

                print_loss_avg = print_loss_total / ((i + 1) * print_every)
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, current_epoch / epochs),
                                             current_epoch, current_epoch / epochs * 100, print_loss_avg))
                if accuracy2 > acc_to_save:
                    acc_to_save = accuracy2
                    Epoch_saved = current_epoch
                    torch.save(model, f'CRNN_Epoch_{Epoch_saved}_Acc_{acc_to_save:.3f}.pth')

            if current_epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / (i + 1) * plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        # showPlot(plot_losses)
    if train_boolean:
        model = torch.load(f'CRNN_Epoch_{Epoch_saved}_Acc_{acc_to_save:.3f}.pth').to(device)
    else:
        model = torch.load(f'CRNN_Epoch_26_Acc_0.972.pth').to(device)
    test_loss = 0

    accuracy = 0
    accuracy2 = 0
    accuracy1 = 0
    model.eval()
    y_true, y_pred = [], []
    # batch_size_2 = features_test.shape[0]
    batch_size_2 = 10
    with torch.no_grad():
        tot_iter_test = int(np.around(features_test.shape[0] / batch_size_2))

        for j in np.arange(0, tot_iter_test):
            inputs = torch.tensor(features_test[j * batch_size_2:j * batch_size_2 + batch_size_2]).type(
                torch.float)
            labels = torch.tensor(target_test[j * batch_size_2:j * batch_size_2 + batch_size_2])

            if inputs.shape[0] == batch_size_2:
                inputs, labels = inputs.to(device), labels.to(device)

                predicted_test = model.forward(inputs)

                batch_loss = criterion(predicted_test, labels)

                test_loss += batch_loss

                sm_of_output = nn.functional.softmax(predicted_test, dim=1)
                # amax = torch.argmax(sm_of_output, dim=1)
                # amax = torch.argmax(sm_of_output, dim=1)
                # equals = torch.eq(labels, amax)
                # accuracy2 += equals.sum()
                # accuracy1 = accuracy_for_each_class(labels.cpu(), amax.cpu())

                amax, _ = stats.mode(torch.argmax(sm_of_output, dim=1).cpu().numpy())
                amax = torch.tensor(amax).repeat(batch_size_2).cuda()
                equals = torch.eq(labels, amax)
                accuracy2 += equals.sum()
                # accuracy1 = accuracy_for_each_class(labels.cpu(), amax.cpu())
                y_true.append(labels.cpu().numpy())
                y_pred.append(amax.cpu().numpy())
    print(
          f"test loss: {test_loss / (j + 1):.3f}.. "
          f"test accuracy: {accuracy2 / ((j + 1) * batch_size_2):.3f}"
          )

    target_names = list(dict_to_label.keys())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    print(classification_report(y_true, y_pred, target_names=target_names))
