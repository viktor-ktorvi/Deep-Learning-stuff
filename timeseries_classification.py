import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy import signal


class Network(nn.Module):
    def __init__(self, signal_len, classes):
        super(Network, self).__init__()

        # Calculating the layer sizes based on the signal length
        kernel_conv1 = 5
        kernel_conv2 = 5
        kernel_pool = 2

        conv1_output_size = (signal_len - kernel_conv1 + 1) / kernel_pool
        self.conv1_output_size = int(conv1_output_size)

        # make sure it's a round number
        assert abs(self.conv1_output_size - conv1_output_size) < 1e-4

        conv2_output_size = (self.conv1_output_size - kernel_conv2 + 1) / kernel_pool
        self.conv2_output_size = int(conv2_output_size)

        assert abs(self.conv2_output_size - conv2_output_size) < 1e-4

        self.conv1 = nn.Conv1d(1, 6, kernel_conv1)
        self.pool = nn.MaxPool1d(kernel_pool)
        self.conv2 = nn.Conv1d(6, 16, kernel_conv2)
        self.fc1 = nn.Linear(16 * self.conv2_output_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(classes))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.conv2_output_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def getSignal(wave, amp, freq, phase, t):
    return amp * wave(2 * np.pi * freq * t + phase)


if __name__ == "__main__":
    # %% Signal parameters
    classes = ["sin", "square", "saw"]
    waves = [np.sin, signal.square, signal.sawtooth]

    Fs = 2000
    signal_len = 100
    t = np.linspace(0, (signal_len - 1) / Fs, signal_len)
    amp_max = 10
    amp_min = 0
    freq_max = 100
    freq_min = 10
    # %% Training parameters
    num_signals = 1000
    num_epochs = 15
    batch_size = 64
    lr = 0.001
    holdout_ratio = 0.7

    train_num = round(holdout_ratio * num_signals)
    test_num = num_signals - train_num
    # %% Generating data
    signal_data = np.zeros((num_signals, signal_len))
    signal_labels = np.zeros((num_signals, 1))

    # make a signal from a random class with random parameters
    for i in range(num_signals):
        chooser = np.random.randint(len(classes))
        signal_labels[i] = chooser

        # uniformally pick parameters
        amp = np.random.rand() * (amp_max - amp_min) + amp_min
        freq = np.random.rand() * (freq_max - freq_min) + freq_min
        phase = np.random.rand() * 2 * np.pi

        # awgn for good measure
        noise_std = 0.1 * amp

        signal_data[i, :] = getSignal(wave=waves[chooser],
                                      amp=amp,
                                      freq=freq,
                                      phase=phase,
                                      t=t).reshape(1, signal_len) + noise_std * np.random.randn(1, signal_len)
    # %% Setting up the data
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # CPU because I have a weak GPU
    print(device)

    tensor_x = torch.tensor(signal_data)
    tensor_y = torch.tensor(signal_labels)
    tensor_y = tensor_y.type(torch.LongTensor)
    dataset = TensorDataset(tensor_x, tensor_y)

    # holdout
    train_set, test_set = torch.utils.data.random_split(dataset, [train_num, test_num])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=test_num)
    # %% Training
    model = Network(signal_len, classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_array = []
    for epoch in range(num_epochs):
        running_loss = 0

        for i, data in enumerate(train_dataloader, 0):
            test_signals, test_labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)

            # reformating the label array to the form that the loss expects
            test_labels = test_labels.view(test_labels.shape[0])
            test_labels = test_labels.type(torch.LongTensor)

            optimizer.zero_grad()

            outputs = model(test_signals.unsqueeze(1))
            loss = criterion(outputs, test_labels)
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
            outputs = model(test_signals.unsqueeze(1))

            _, predicted = torch.max(outputs, 1)

    predicted = predicted.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy().reshape(test_num)

    results = predicted == test_labels

    print("\nAccurracy = %.3f %%" % (np.sum(results) / len(results) * 100))

    # %% Visually checking results
    check_num = 3
    check_set, _ = torch.utils.data.random_split(test_set, [check_num, test_num - check_num])

    check_dataloader = DataLoader(check_set, batch_size=check_num)

    with torch.no_grad():
        plt.figure()
        for data in check_dataloader:
            test_signals, test_labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)
            outputs = model(test_signals.unsqueeze(1))

            _, predicted = torch.max(outputs, 1)
            predicted = predicted.detach().cpu().numpy()

        for i, test_signal in enumerate(test_signals):
            plt.plot(test_signal, label="predicted " + classes[predicted[i]])

        plt.title("Samles and predictions")
        plt.xlabel("n [sample]")
        plt.ylabel("signal [num]")
        plt.legend()
        plt.show()
