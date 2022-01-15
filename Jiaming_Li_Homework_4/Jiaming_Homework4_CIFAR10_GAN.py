# Jiaming Li
# University of Ottawa
# CSI 5340 Intro to Deep Learning and Reinforcement Learning
# Homework 4
# CIFAR10 Generative Adversarial Network

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import math
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# CIFAR10 Dataset
def get_dataset_CIFAR10(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(root = 'D:/李佳明/University_of_Ottawa/Fall_2021/CSI_5340/Jiaming_Li_Homework_4',
                                     train = True,
                                     transform = transform,
                                     download = True)
    test_dataset = datasets.CIFAR10(root = 'D:/李佳明/University_of_Ottawa/Fall_2021/CSI_5340/Jiaming_Li_Homework_4',
                                    train = False,
                                    transform = transform)

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

    return train_dataset, test_dataset, train_loader, test_loader

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

def Discriminator_train(x, input_dim, batch_size):
    Discriminator.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, cifar10_dim), torch.ones(batch_size, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    Discriminator_output = Discriminator(x_real)
    Discriminator_real_loss = criterion(Discriminator_output, y_real)
    # Discriminator_real_score = Discriminator_output

    # train discriminator on facke
    z = Variable(torch.randn(batch_size, input_dim).to(device))
    x_fake, y_fake = Generator(z), Variable(torch.zeros(batch_size, 1).to(device))

    Discriminator_output = Discriminator(x_fake)
    Discriminator_fake_loss = criterion(Discriminator_output, y_fake)
    # Discriminator_fake_score = Discriminator_output

    # gradient backprop & optimize ONLY D's parameters
    Discriminator_loss = Discriminator_real_loss + Discriminator_fake_loss
    Discriminator_loss.backward()
    Discriminator_optimizer.step()

    return  Discriminator_loss.data.item()

def Generator_train(x, input_dim, batch_size):
    Generator.zero_grad()

    z = Variable(torch.randn(batch_size, input_dim).to(device))
    y = Variable(torch.ones(batch_size, 1).to(device))

    Generator_output = Generator(z)
    Discriminator_output = Discriminator(Generator_output)
    Generator_loss = criterion(Discriminator_output, y)

    # gradient backprop & optimize ONLY G's parameters
    Generator_loss.backward()
    Generator_optimizer.step()

    return Generator_loss.data.item()

def training_model(num_epochs, train_loader):
    Generator_losses_list = []
    Discriminator_losses_list = []
    JSD_losses_list = []
    for epoch in range(num_epochs):
        loss_d_list = []
        loss_g_list = []
        jsd_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_loader):
            loss_d_list.append(Discriminator_train(x, input_dim, batch_size))
            loss_g_list.append(Generator_train(x, input_dim, batch_size))
            jsd_loss += 0.5 * (-Discriminator_train(x, input_dim, batch_size) + math.log(4))

        Generator_losses = torch.mean(torch.FloatTensor(loss_d_list))
        Discriminator_losses = torch.mean(torch.FloatTensor(loss_g_list))
        Generator_losses_list.append(float(Generator_losses))
        Discriminator_losses_list.append(float(Discriminator_losses))
        JSD_losses_list.append(jsd_loss)

        print(f'[Epoch]: {epoch + 1} [Generator Loss]: {Generator_losses:.4f} '
              f'[Discriminator Loss]: {Discriminator_losses:.4f} [JSD Loss]: {jsd_loss}')

    print('Generator_losses_list = ', Generator_losses_list)
    print('Discriminator_losses_list = ', Discriminator_losses_list)
    print('JSD_losses_list = ', JSD_losses_list)

    return Generator_losses_list, Discriminator_losses_list, JSD_losses_list

if __name__ == "__main__":
    batch_size = 100
    input_dim = 100
    train_dataset, test_dataset, train_loader, test_loader = get_dataset_CIFAR10(batch_size)

    # train_dataset = torch.Tensor(train_dataset)
    # train_dataset = [t.torch.Tensor() for t in train_dataset]
    # print(type(train_dataset.data.size(1)), train_dataset.data.size(1))
    cifar10_dim = 3 * 32 * 32

    Generator = Generator(g_input_dim = input_dim, g_output_dim = cifar10_dim).to(device)
    Discriminator = Discriminator(cifar10_dim).to(device)

    # loss
    criterion = nn.BCELoss()
    # optimizer
    learning_rate = 0.0002
    Generator_optimizer = optim.Adam(Generator.parameters(), lr = learning_rate)
    Discriminator_optimizer = optim.Adam(Discriminator.parameters(), lr = learning_rate)

    num_epochs = 100

    epochs = [i for i in range(1, num_epochs + 1)]
    Generator_losses_list, Discriminator_losses_list, JSD_losses_list = training_model(num_epochs, train_loader)

    # Generator & Discriminator Loss Plot
    plt.plot(epochs, Generator_losses_list, color = 'tomato', label = 'Generator Loss')
    plt.plot(epochs, Discriminator_losses_list, color = 'limegreen', label = 'Discriminator Loss')
    plt.title('Generator and Discriminator Loss of CIFAR10 Dataset (GAN)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc = "upper right")
    plt.show()

    # JSD for GAN (Jensen- Shannon Divergence)
    plt.plot(epochs, JSD_losses_list, color = 'blue')
    plt.title('JSD of CIFAR10 Dataset (GAN)')
    plt.xlabel('Epoch')
    plt.ylabel('JSD Value')
    plt.show()

