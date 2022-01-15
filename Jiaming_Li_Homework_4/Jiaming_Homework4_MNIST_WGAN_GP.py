# Jiaming Li
# University of Ottawa
# CSI 5340 Intro to Deep Learning and Reinforcement Learning
# Homework 4
# MNIST Wasserstein Generative Adversarial Network - Gradient Penalty

import torch
import torchvision
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def get_dataset_MNIST(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root = 'D:/李佳明/University_of_Ottawa/Fall_2021/CSI_5340/Jiaming_Li_Homework_4',
                                   train = True,
                                   transform = transform,
                                   download = True)
    test_dataset = datasets.MNIST(root = 'D:/李佳明/University_of_Ottawa/Fall_2021/CSI_5340/Jiaming_Li_Homework_4',
                                  train = False,
                                  transform = transform)

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader

def to_onehot(x, num_classes=10):
    assert isinstance(x, int) or isinstance(x, (torch.LongTensor, torch.cuda.LongTensor))
    if isinstance(x, int):
        c = torch.zeros(1, num_classes).long()
        c[0][x] = 1
    else:
        x = x.cpu()
        c = torch.LongTensor(x.size(0), num_classes)
        c.zero_()
        c.scatter_(1, x, 1) # dim, index, src value
    return c

# def get_sample_image(G, n_noise=100):
#     img = np.zeros([280, 280])
#     for j in range(10):
#         c = torch.zeros([10, 10]).to(device)
#         c[:, j] = 1
#         z = torch.randn(10, n_noise).to(device)
#         y_hat = G(z,c).view(10, 28, 28)
#         result = y_hat.cpu().data.numpy()
#         img[j*28:(j+1)*28] = np.concatenate([x for x in result], axis=-1)
#     return img

class Discriminator(nn.Module):
    def __init__(self, in_channel=1, input_size=784, condition_size=10, num_classes=1):
        super(Discriminator, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_size+condition_size, 784),
            nn.LeakyReLU(0.2),
        )
        self.conv = nn.Sequential(
            # 28 -> 14
            nn.Conv2d(in_channel, 512, 3, stride=2, padding=1, bias=False),
            #             nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 14 -> 7
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),
            #             nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 7 -> 4
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            #             nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4),
        )
        self.fc = nn.Sequential(
            # reshape input, 128 -> 1
            nn.Linear(128, 1),
        )

    def forward(self, x, c=None):
        # x: (N, 1, 28, 28), c: (N, 10)
        x, c = x.view(x.size(0), -1), c.float() # may not need
        v = torch.cat((x, c), 1) # v: (N, 794)
        y_ = self.transform(v) # (N, 784)
        y_ = y_.view(y_.shape[0], 1, 28, 28) # (N, 1, 28, 28)
        y_ = self.conv(y_)
        y_ = y_.view(y_.size(0), -1)
        y_ = self.fc(y_)
        return y_

class Generator(nn.Module):
    def __init__(self, input_size=100, condition_size=10):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size+condition_size, 4*4*512),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            # input: 4 by 4, output: 7 by 7
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, bias=False),
            #             nn.BatchNorm2d(256),
            nn.ReLU(),
            # input: 7 by 7, output: 14 by 14
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            #             nn.BatchNorm2d(128),
            nn.ReLU(),
            # input: 14 by 14, output: 28 by 28
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, c):
        # x: (N, 100), c: (N, 10)
        x, c = x.view(x.size(0), -1), c.float() # may not need
        v = torch.cat((x, c), 1) # v: (N, 110)
        y_ = self.fc(v)
        y_ = y_.view(y_.size(0), 512, 4, 4)
        y_ = self.conv(y_) # (N, 28, 28)
        return y_


def training_model(step, num_epochs, train_loader):
    Generator_losses_list = []
    Discriminator_losses_list = []
    EMD_value_list = []
    for epoch in range(num_epochs):
        emd_value = 0.0
        for idx, (images, labels) in enumerate(train_loader):
            Discriminator.zero_grad()
            x = images.to(device)
            y = labels.view(batch_size, 1)
            y = to_onehot(y).to(device)
            z = torch.randn(batch_size, n_noise).to(device)
            x_fake = Generator(z, y)

            # Gradient Penalty
            eps = torch.rand(batch_size, 1, 1, 1).to(device)
            x_penalty = eps * x + (1 - eps) * x_fake
            x_penalty = x_penalty.view(x_penalty.size(0), -1)
            p_outputs = Discriminator(x_penalty, y)
            xp_grad = autograd.grad(outputs = p_outputs, inputs = x_penalty, grad_outputs = Discriminator_labels,
                                create_graph = True, retain_graph = True, only_inputs = True)
            grad_penalty = p_coeff * torch.mean(torch.pow(torch.norm(xp_grad[0], 2, 1) - 1, 2))

            # Wasserstein loss
            x_outputs = Discriminator(x, y)
            z_outputs = Discriminator(x_fake, y)
            Discriminator_x_loss = torch.mean(x_outputs)
            Discriminator_z_loss = torch.mean(z_outputs)
            Discriminator_loss = Discriminator_z_loss - Discriminator_x_loss + grad_penalty

            Discriminator_loss.backward()
            Discriminator_opt.step()
            emd_value += (-Discriminator_loss.item())
            if step % n_critic == 0:
                Discriminator.zero_grad()
                Generator.zero_grad()
                # Training Generator
                z = torch.randn(batch_size, n_noise).to(device)
                z_outputs = Discriminator(Generator(z, y), y)
                Generator_loss = -torch.mean(z_outputs)
                Generator_loss.backward()
                Generator_opt.step()

            if step % 1000 == 0:
                print('Epoch: {}/{}, Discriminator Loss: {}, Generator Loss: {}, [EMD Value]: {}'.format
                      (epoch, num_epochs, Discriminator_loss.item(), Generator_loss.item(), emd_value))
                Generator_losses_list.append(Generator_loss.item())
                Discriminator_losses_list.append(Discriminator_loss.item())
                EMD_value_list.append(emd_value)

            step += 1

    print('Generator_losses_list = ', Generator_losses_list)
    print('Discriminator_losses_list = ', Discriminator_losses_list)
    print('EMD_value_list = ', EMD_value_list)
    return Generator_losses_list, Discriminator_losses_list, EMD_value_list

if __name__ == "__main__":
    batch_size = 64
    learning_rate = 0.0001
    num_epochs = 50
    step = 0
    n_noise = 100
    p_coeff = 10
    n_critic = 5

    Discriminator = Discriminator().to(device)
    Generator = Generator().to(device)
    train_loader, test_loader = get_dataset_MNIST(batch_size)

    Discriminator_opt = torch.optim.Adam(Discriminator.parameters(), lr = learning_rate, betas=(0., 0.9))
    Generator_opt = torch.optim.Adam(Generator.parameters(), lr = learning_rate, betas=(0., 0.9))

    Discriminator_labels = torch.ones([batch_size, 1]).to(device)
    Discriminator_fakes = torch.zeros([batch_size, 1]).to(device)

    Generator_losses_list, Discriminator_losses_list, EMD_value_list = training_model(step, num_epochs, train_loader)

    epochs = [i for i in range(1, num_epochs + 1)]

    # Generator & Discriminator Loss Plot
    plt.plot(epochs, Generator_losses_list, color = 'tomato', label = 'Generator Loss')
    plt.plot(epochs, Discriminator_losses_list, color = 'limegreen', label = 'Discriminator Loss')
    plt.title('Generator and Discriminator Loss of MNIST Dataset (WGAN-GP)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc = "upper right")
    plt.show()

    # EMD for WGAN (Earth Mover's Distance)
    plt.plot(epochs, EMD_value_list, color = 'blue')
    plt.title('EMD of MNIST Dataset (WGAN-GP)')
    plt.xlabel('Epoch')
    plt.ylabel('EMD Value')
    plt.show()
