import torch
import torch.nn as nn
import torch.nn.functional as F

print('cuda', torch.cuda.is_available())


class MusicClassificationCRNN(nn.Module):
    def __init__(self, cfg):
        super(MusicClassificationCRNN, self).__init__()
        num_class = cfg.MODEL.NUM_CLASSES

        self.np_layers = 4
        self.np_filters = [64, 128, 128, 128]
        self.kernel_size = (3, 3)

        self.pool_size = [(2, 2), (4, 2)]

        self.channel_axis = 1
        self.frequency_axis = 2
        self.time_axis = 3

        # self.h0 = torch.zeros((1, 16, 32)).to(device)

        self.bn0 = nn.BatchNorm2d(num_features=self.channel_axis)
        self.bn1 = nn.BatchNorm2d(num_features=self.np_filters[0])
        self.bn2 = nn.BatchNorm2d(num_features=self.np_filters[1])
        self.bn3 = nn.BatchNorm2d(num_features=self.np_filters[2])
        self.bn4 = nn.BatchNorm2d(num_features=self.np_filters[3])

        self.conv1 = nn.Conv2d(1, self.np_filters[0], kernel_size=self.kernel_size)
        self.conv2 = nn.Conv2d(self.np_filters[0], self.np_filters[1], kernel_size=self.kernel_size)
        self.conv3 = nn.Conv2d(self.np_filters[1], self.np_filters[2], kernel_size=self.kernel_size)
        self.conv4 = nn.Conv2d(self.np_filters[2], self.np_filters[3], kernel_size=self.kernel_size)

        self.max_pool_2_2 = nn.MaxPool2d(self.pool_size[0])
        self.max_pool_4_2 = nn.MaxPool2d(self.pool_size[1])

        self.drop_01 = nn.Dropout(0.1)
        self.drop_03 = nn.Dropout(0.3)

        self.gru1 = nn.GRU(input_size=128, hidden_size=32, batch_first=True)
        self.gru2 = nn.GRU(input_size=32, hidden_size=32, batch_first=True)

        self.activation = nn.ELU()

        self.dense = nn.Linear(32, num_class)
        # self.softmax = nn.Softmax(dim=1)

        # self.noise = GaussianNoise()

    def forward(self, x):
        # x [16, 1, 128,1024]
        x = self.bn0(x)
        # x [16, 1, 128,1024]
        x = F.pad(x, (0, 0, 2, 1))
        # x [16, 1, 131,1024]
        x = self.conv1(x)
        # x [16, 64, 129,1022]
        x = self.activation(x)
        # x [16, 64, 129,1022]
        x = self.bn1(x)
        # x [16, 64, 129,1022]
        x = self.max_pool_2_2(x)
        # x [16, 64, 64,511]
        x = self.drop_01(x)
        # x [16, 64, 64,511]
        x = F.pad(x, (0, 0, 2, 1))
        # x [16, 64, 67,511]
        x = self.conv2(x)
        # x [16, 128, 65,509]
        x = self.activation(x)
        # x [16, 128, 65,509]
        x = self.bn2(x)
        # x [16, 128, 65,509]
        x = self.max_pool_4_2(x)
        # x [16, 128, 16,254]
        x = self.drop_01(x)
        # x [16, 128, 16,254]
        x = F.pad(x, (0, 0, 2, 1))
        # x [16, 128, 19,254]
        x = self.conv3(x)
        # x [16, 128, 17,252]
        x = self.activation(x)
        # x [16, 128, 17,252]
        x = self.bn3(x)
        # x [16, 128, 17,252]
        x = self.max_pool_4_2(x)
        # x [16, 128, 4,126]
        x = self.drop_01(x)
        # x [16, 128, 4,126]
        x = F.pad(x, (0, 0, 2, 1))
        # x [16, 128, 7,126]
        x = self.conv4(x)
        # x [16, 128, 5,124]
        x = self.activation(x)
        # x [16, 128, 5,124]
        x = self.bn4(x)
        # x [16, 128, 5,124]
        x = self.max_pool_4_2(x)
        # x [16, 128, 1,62]
        x = self.drop_01(x)
        # x [16, 128, 1,62]

        x = x.permute(0, 3, 1, 2)
        # x [16, 62, 128,1]
        resize_shape = list(x.shape)[2] * list(x.shape)[3]
        # x [16, 62,128,1], reshape size is 128
        x = torch.reshape(x, (list(x.shape)[0], list(x.shape)[1], resize_shape))
        # x [16, 62, 128]
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
        h0 = torch.zeros((1, x.size(0), 32)).to(device)
        x, h1 = self.gru1(x, h0)
        # x [16, 62, 32]
        x, _ = self.gru2(x, h1)
        # x [16, 62, 32]
        x = x[:, -1, :]
        # x [16,32]
        x = self.dense(x)
        # x [16,10]
        # x = self.softmax(x)
        # x [16, 10]
        # x = torch.argmax(x, 1)
        return x


def add_noise_to_weights(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * 0.1)
