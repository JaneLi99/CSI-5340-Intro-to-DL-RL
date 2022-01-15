# Jiaming Li
# University of Ottawa
# CSI 5340 Intro to Deep Learning and Reinforcement Learning
# Homework 4
# CIFAR10 Wasserstein Generative Adversarial Network

import torch
from torchtext import datasets
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

    return train_dataset, train_loader, test_dataset, test_loader

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_vector_dim, 32 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(32 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(32 * 8, 32 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( 32 * 4, 32 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( 32 * 2, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 2, 32 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 4, 1, 4, 1, 0, bias=False),
        )
    def forward(self, input):
        return self.main(input)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_discriminator(discriminator, optimizer, data_real, data_fake):
    discriminator.train()
    b_size = data_real.size(0) # Get the batch size of data
    optimizer.zero_grad()
    output_real = discriminator(data_real).view(-1) # real data
    output_fake = discriminator(data_fake).view(-1) # fake data
    Discriminator_loss = -(torch.mean(output_real) - torch.mean(output_fake))
    Discriminator_loss.backward()
    optimizer.step()

    for p in discriminator.parameters():
        p.data.clamp_(-weight_clip, weight_clip)

    return Discriminator_loss

def train_generator(generator, discriminator, optimizer, data_fake):
    generator.train()
    b_size = data_fake.size(0)

    optimizer.zero_grad()
    output = discriminator(data_fake).view(-1)
    loss_G = -torch.mean(output)
    loss_G.backward()
    optimizer.step()

    return loss_G

# Generate batch of latent vectors
def create_noise(sample_size, latent_vector_dim):
    return torch.randn(sample_size, latent_vector_dim, 1, 1, device=device)

def training_model(train_loader, train_dataset, num_epochs):
    generator = Generator().to(device)
    generator.apply(weights_init)
    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)

    optim_Generator = optim.RMSprop(generator.parameters(), lr = learning_rate)
    optim_Discriminator = optim.RMSprop(discriminator.parameters(), lr = learning_rate)
    Generator_losses_list = []
    Discriminator_losses_list = []
    emd_list = []
    epoch_list = []

    fixed_noise = torch.randn(64, latent_vector_dim, 1, 1, device=device)

    for epoch in range(num_epochs):
        loss_Generator = 0.0
        loss_Discriminator = 0.0
        em_distance = 0.0

        for idx, data in tqdm(enumerate(train_loader), total = int(len(train_dataset) / train_loader.batch_size)):
            image, _ = data
            image = image.to(device)
            b_size = len(image)

            for step in range(k):
                data_fake = generator(create_noise(b_size, latent_vector_dim)).detach()
                data_real = image
                # train the discriminator
                loss_Discriminator1= train_discriminator(discriminator, optim_Discriminator, data_real, data_fake)
                loss_Discriminator += loss_Discriminator1
                em_distance += (-loss_Discriminator1)

            data_fake = generator(create_noise(b_size, latent_vector_dim))
            # train the generator
            loss_Generator1 = train_generator(generator, discriminator, optim_Generator, data_fake)
            loss_Generator += loss_Generator1

        # Check how the generator is doing by saving G's output on fixed_noise
        epoch_loss_Generator = loss_Generator / idx
        epoch_loss_Discriminator = loss_Discriminator / (k * idx)
        epoch_emd = em_distance / (k * idx)

        Generator_losses_list.append(epoch_loss_Generator)
        Discriminator_losses_list.append(epoch_loss_Discriminator)
        epoch_list.append(epoch + 1)
        emd_list.append(epoch_emd)

        print(f"Epoch {epoch + 1} of {num_epochs}")
        print(f"Generator loss: {epoch_loss_Generator:.4f}, Discriminator loss: {epoch_loss_Discriminator:.4f}, EMD: {epoch_emd:.4f}")
    print('Generator_losses_list = ', Generator_losses_list)
    print('Discriminator_losses_list = ', Discriminator_losses_list)
    print('EMD_list = ', emd_list)
    return Generator_losses_list, Discriminator_losses_list, emd_list


if __name__=="__main__":
    SEED = 1234
    torch.manual_seed(SEED)

    batch_size = 128
    latent_vector_dim = 128
    num_epochs = 50
    learning_rate = 5e-5
    sample_size = 64
    k = 5
    weight_clip = 0.01

    train_dataset, train_loader, test_dataset, test_loader = get_dataset_CIFAR10(batch_size)
    Generator_losses_list, Discriminator_losses_list, EMD_list = training_model(train_loader, train_dataset, num_epochs)

    epochs = [i for i in range(1, num_epochs + 1)]
    plt.plot(epochs, Generator_losses_list, color = 'tomato', label ='Generator loss' )
    plt.plot(epochs, Discriminator_losses_list, color = 'limegreen', label = 'Discriminator loss')
    plt.title('Generator and Discriminator Loss of CIFAR10 Dataset (WGAN)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc = "upper right")
    plt.show()

    plt.plot(epochs, EMD_list, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('EMD Value')
    plt.title('EMD of CIFAR10 Dataset (WGAN)')



