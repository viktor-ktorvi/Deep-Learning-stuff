from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn


# %%
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2, 10)
        self.output = nn.Linear(10, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)

        return x


# %%

if __name__ == '__main__':
    plt.close('all')
    num = 5000
    my_class1 = np.random.randn(2, num)
    # my_class2 = np.array([[4], [1]]) + np.random.randn(2, num)
    # helper_class = np.array([[-4], [1]]) + np.random.randn(2, num)
    # my_class2 = np.hstack((my_class2, helper_class))
    rho = 4
    theta = np.random.randn(1, num) + np.pi / 2
    my_class2 = (rho + 0.2 * np.random.randn(2, num)) * np.vstack((np.cos(theta), np.sin(theta)))

    total_num = my_class1.shape[1] + my_class2.shape[1]

    markersize = 1
    plt.figure()
    plt.plot(my_class1[0, :], my_class1[1, :], 'og', markersize=markersize)
    plt.plot(my_class2[0, :], my_class2[1, :], 'oy', markersize=markersize)

    plt.axis('equal')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Classes")
    plt.show()
    # %%
    all_samples = np.hstack((my_class1, my_class2))
    labels = np.hstack((np.zeros(my_class1.shape[1]), np.ones(my_class2.shape[1])))
    plt.figure()
    plt.plot(all_samples[0, :], all_samples[1, :], 'om', markersize=markersize)

    plt.axis('equal')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("All samples")
    plt.show()
    # %%
    tensor_x = torch.Tensor(all_samples.T)
    tensor_y = torch.Tensor(labels.T)
    tensor_y = tensor_y.type(torch.LongTensor)

    train_num = round(0.7 * total_num)
    test_num = total_num - train_num

    batch_size = 64

    my_dataset = TensorDataset(tensor_x, tensor_y)
    train_set, test_set = torch.utils.data.random_split(my_dataset, [train_num, test_num])
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=test_num)

    # %%
    model = Network()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    epochs = 10

    for e in range(epochs):
        running_loss = 0
        for x, y in train_dataloader:
            optimizer.zero_grad()
            output = model.forward(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(running_loss)
    print("Training complete.")
    for x, y in test_dataloader:
        test_output = model.forward(x)
        test_loss = criterion(test_output, y)
    print("Test complete")

    x_np = x.detach().numpy()
    test_output_np = test_output.detach().numpy()
    # %%
    plt.figure()
    plt.plot(x_np[test_output_np[:, 0] > test_output_np[:, 1]][:, 0],
             x_np[test_output_np[:, 0] > test_output_np[:, 1]][:, 1], 'bo', markersize=markersize)
    plt.plot(x_np[test_output_np[:, 0] <= test_output_np[:, 1]][:, 0],
             x_np[test_output_np[:, 0] <= test_output_np[:, 1]][:, 1], 'ro', markersize=markersize)

    plt.axis('equal')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title("Test")
    plt.show()
