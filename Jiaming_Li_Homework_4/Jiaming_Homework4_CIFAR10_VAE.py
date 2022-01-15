# Jiaming Li
# University of Ottawa
# CSI 5340 Intro to Deep Learning and Reinforcement Learning
# Homework 4
# CIFAR10 Variational Auto-Encoder

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# CIFAR10 Dataset
def get_dataset_CIFAR10(batch_size):
    train_dataset = datasets.CIFAR10(root='D:/李佳明/University_of_Ottawa/Fall_2021/CSI_5340/Jiaming_Li_Homework_4',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)
    test_dataset = datasets.CIFAR10(root='D:/李佳明/University_of_Ottawa/Fall_2021/CSI_5340/Jiaming_Li_Homework_4',
                                    train=False,
                                    transform=transforms.ToTensor())

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

    return train_dataset, test_dataset, train_loader, test_loader

# Define model architecture and reconstruction loss
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # 32 * 32
            nn.Linear(1024, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, d * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, 1024),
            nn.Sigmoid(),
        )

    def reparameteries(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, y):
        mu_logvar = self.encoder(y.view(-1, 1024)).view(-1, 2, d)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameteries(mu, logvar)
        return self.decoder(z), mu, logvar

def loss_function(y_hat, y, mu, logvar):
    BCE = F.binary_cross_entropy(
        y_hat, y.view(-1, 1024), reduction = 'sum'
    )
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

    return BCE, KLD, BCE + KLD

def training_model(num_epochs):
    train_loss_list = []
    test_loss_list = []
    BCE_train_loss_list = []
    BCE_test_loss_list = []
    KLD_train_loss_list = []
    KLD_test_loss_list = []
    train_dataset, test_dataset, train_loader, test_loader = get_dataset_CIFAR10(batch_size)
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        BCE_train_loss = 0
        KLD_train_loss = 0
        for y, _ in train_loader:
            y = y.to(device)

            y_hat, mu, logvar = model(y)
            BCE_loss, KLD_loss, loss = loss_function(y_hat, y, mu, logvar)
            train_loss += loss.item()
            BCE_train_loss += BCE_loss.item()
            KLD_train_loss += KLD_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Testing
        test_loss = 0
        BCE_test_loss = 0
        KLD_test_loss = 0
        with torch.no_grad():
            model.eval()
            for y, x in test_loader:
                y = y.to(device)

                # forward
                y_hat, mu, logvar = model(y)
                BCE_loss, KLD_loss, loss = loss_function(y_hat, y, mu, logvar)
                test_loss += loss.item()
                BCE_test_loss += BCE_loss.item()
                KLD_test_loss += KLD_loss.item()

        train_loss = train_loss / len(train_loader.dataset)
        train_loss_list.append(train_loss)
        test_loss = test_loss / len(test_loader.dataset)
        test_loss_list.append(test_loss)
        BCE_train_loss = BCE_train_loss / len(train_loader.dataset)
        BCE_train_loss_list.append(BCE_train_loss)
        BCE_test_loss = BCE_test_loss/len(test_loader.dataset)
        BCE_test_loss_list.append(BCE_test_loss)
        KLD_train_loss = KLD_train_loss / len(train_loader.dataset)
        KLD_train_loss_list.append(KLD_train_loss)
        KLD_test_loss = KLD_test_loss/len(test_loader.dataset)
        KLD_test_loss_list.append(KLD_test_loss)

        print(f'[Epoch]: {epoch} [Train Loss]: {train_loss:.4f} [Test Loss]: {test_loss:.4f}')

    print('train_loss_list = ', train_loss_list)
    print('test_loss_list = ', test_loss_list)
    print('BCE_train_loss_list = ', BCE_train_loss_list)
    print('BCE_test_loss_list = ', BCE_test_loss_list)
    print('KLD_train_loss_list = ', KLD_train_loss_list)
    print('KLD_test_loss_list = ', KLD_test_loss_list)

    return train_loss_list, test_loss_list, BCE_train_loss_list, BCE_test_loss_list, KLD_train_loss_list, KLD_test_loss_list

if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    d = 20
    batch_size = 256
    learning_rate = 0.001
    num_epochs = 100

    model = VAE().to(device)
    # Configure the optimizer
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    train_loss_list, test_loss_list, BCE_train_loss_list, BCE_test_loss_list, KLD_train_loss_list, KLD_test_loss_list = training_model(num_epochs)
    epochs = [i for i in range(1, num_epochs + 1)]

    # Training & Testing Loss Plot
    plt.plot(epochs, train_loss_list, color = 'tomato', label = 'Train Loss')
    plt.plot(epochs, test_loss_list, color = 'limegreen', label = 'Test Loss')
    plt.title('Train and Test Loss of CIFAR10 Dataset (VAE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc = "upper right")
    plt.show()

    # KLD loss
    plt.plot(epochs, KLD_train_loss_list, color = 'steelblue', label = 'KLD Train Loss')
    plt.plot(epochs, KLD_test_loss_list, color = 'orange', label = 'KLD Test Loss')
    plt.title('KLD Train and Test Loss of CIFAR10 Dataset (VAE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc = "upper right")
    plt.show()

    # BCE loss
    plt.plot(epochs, BCE_train_loss_list, color = 'steelblue', label = 'BCE Train Loss')
    plt.plot(epochs, BCE_test_loss_list, color = 'orange', label = 'BCE Test Loss')
    plt.title('BCE Train and Test Loss of CIFAR10 Dataset (VAE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc = "upper right")
    plt.show()

