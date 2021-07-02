import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy import signal

from timeseries_classification import getSignal


def calcConv1dOutSize(in_sz, kernel_sz):
    out = in_sz - kernel_sz + 1
    assert out > 0
    # TODO throw exception

    return out


def calcPoolOutSize(in_sz, pool_sz):
    out_float = in_sz / pool_sz
    out = int(out_float)

    assert abs(out - out_float) < 1e-4
    # TODO throw exception

    return out


def calcConv1dPoolOutSize(in_sz, kernel_sz, pool_sz):
    return calcPoolOutSize(calcConv1dOutSize(in_sz, kernel_sz), pool_sz)


class Network(nn.Module):
    def __init__(self, signal_len):
        super(Network, self).__init__()

        kernel_pool = 2
        kernel_conv1 = 5
        kernel_conv2 = 5
        out_conv1 = calcConv1dPoolOutSize(signal_len, kernel_conv1, kernel_pool)
        out_conv2 = calcConv1dPoolOutSize(out_conv1, kernel_conv2, kernel_pool)

        # TODO configure padding so that the edges look better
        self.pool = nn.MaxPool1d(kernel_pool)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=20, kernel_size=kernel_conv1)
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=50, kernel_size=kernel_conv2)

        stride = 2
        out_deconv1 = round(signal_len / 2)
        # TODO foo for kernel size calc
        # Lout = (Lin - 1) * stride - 2*padding + dilation*(kernel_size - 1) + output_padding + 1
        kernel_deconv1 = out_deconv1 - stride * (out_conv2 - 1)
        kernel_deconv2 = signal_len - stride * (out_deconv1 - 1)

        self.deconv1 = nn.ConvTranspose1d(in_channels=50, out_channels=35, stride=stride, kernel_size=kernel_deconv1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=35, out_channels=1, stride=stride, kernel_size=kernel_deconv2)

    def forward(self, x):

        # encoder
        # 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # decoder
        # 1
        x = self.deconv1(x)
        x = F.relu(x)

        # 2
        x = self.deconv2(x)

        return x


if __name__ == "__main__":
    # %% Signal parameters
    classes = ["sin"]  # ["sin", "square", "saw"]
    waves = [np.sin]  # [np.sin, signal.square, signal.sawtooth]

    Fs = 2000
    signal_len = 200
    t = np.linspace(0, (signal_len - 1) / Fs, signal_len)
    amp_max = 10
    amp_min = 0
    freq_max = 100
    freq_min = 10

    noise_std_percent = 0.1
    # %% Training parameters
    num_signals = 10000
    num_epochs = 12
    batch_size = 64
    lr = 0.004
    holdout_ratio = 0.7

    train_num = round(holdout_ratio * num_signals)
    test_num = num_signals - train_num

    # %% Generate data

    signal_data = np.zeros((num_signals, signal_len))
    signal_labels = np.zeros((num_signals, signal_len))

    # make a signal from a random class with random parameters
    for i in range(num_signals):
        chooser = np.random.randint(len(classes))

        # uniformally pick parameters
        amp = np.random.rand() * (amp_max - amp_min) + amp_min
        freq = np.random.rand() * (freq_max - freq_min) + freq_min
        phase = np.random.rand() * 2 * np.pi

        # awgn for good measure
        noise_std = noise_std_percent * amp

        signal_labels[i, :] = getSignal(wave=waves[chooser],
                                        amp=amp,
                                        freq=freq,
                                        phase=phase,
                                        t=t).reshape(1, signal_len)

        signal_data[i, :] = signal_labels[i, :] + noise_std * np.random.randn(1, signal_len)

    data_std = np.std(signal_data)
    # %% Visualizing the data
    plt.figure()
    plt.title("No noise VS noise")
    for i in range(1):
        idx = np.random.randint(num_signals)
        plt.plot(signal_labels[idx, :], label="signal")
        plt.plot(signal_data[idx, :], label="signal + noise")
    plt.xlabel("n [sample]")
    plt.ylabel("signal [num]")
    plt.legend()
    plt.show()

    # %% Setting up the data
    device = torch.device("cpu")  # CPU because I have a weak GPU
    print(device)

    dataset = TensorDataset(torch.tensor(signal_data), torch.tensor(signal_labels))

    # holdout
    train_set, test_set = torch.utils.data.random_split(dataset, [train_num, test_num])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=test_num)
    # %% Training
    torch.manual_seed(26)
    model = Network(signal_len)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_array = []
    for epoch in range(num_epochs):
        running_loss = 0

        for i, data in enumerate(train_dataloader, 0):
            test_signals, test_labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)

            optimizer.zero_grad()

            outputs = model(test_signals.unsqueeze(1) / data_std) * data_std
            loss = criterion(outputs, test_labels.view(outputs.shape))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("epoch: %d\tloss: %0.10f" % (epoch, running_loss))
        loss_array.append(running_loss)

    plt.figure()
    plt.title("Loss")
    plt.xlabel("epoch [num]")
    plt.ylabel("loss [num]")
    plt.plot(loss_array)
    plt.show()
    # %% Testing

    # pass the whole test set thorugh the model and check outputs
    with torch.no_grad():
        for data in test_dataloader:
            test_signals, test_labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)
            outputs = model(test_signals.unsqueeze(1) / data_std) * data_std

            mse = criterion(outputs, test_labels.view(outputs.shape))
            print("test MSE = %.3f" % mse)
    # %% Visually checking results

    plt.figure()
    plt.title("Noisy VS denoised")
    for i in range(1):
        idx = np.random.randint(test_num)
        plt.plot(test_signals[idx, :], label="noisy")
        plt.plot(outputs[idx, :].T, label="denoised")
        plt.plot(test_labels[idx, :], label="ground truth")
    plt.xlabel("n [sample]")
    plt.ylabel("signal [num]")
    plt.legend()
    plt.show()

