# Jiaming Li
# University of Ottawa
# CSI 5340 Intro to Deep Learning and Reinforcement Learning
# Homework 2 Question 3
# Soft-max Regression & Multilayer Perception

import time
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dataset(batch_size):
    # MNIST Dataset
    train_dataset = datasets.MNIST(root = 'D:/李佳明/University_of_Ottawa/Fall_2021/CSI_5340/Jiaming_Li_Homework_2',
                                   train = True,
                                   transform = transforms.ToTensor(),
                                   download = True)
    test_dataset = datasets.MNIST(root = 'D:/李佳明/University_of_Ottawa/Fall_2021/CSI_5340/Jiaming_Li_Homework_2',
                                  train = False,
                                  transform = transforms.ToTensor())

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

    # Checking the dataset
    # for images, labels in train_loader:
    #     print('Image batch dimensions: ', images.shape)
    #     print('Image batch dimensions: ', labels.shape)
    #     break

    # images[0].shape
    # images[0].view(-1, 28*28).shape
    # images.view(-1, 28*28).shape # n * m

    return train_dataset, test_dataset, train_loader, test_loader

# Soft-max Regression Model
class SoftmaxRegression(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)

        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()

    def forward(self, x):
        logits = self.linear(x)
        #probas = F.softmax(logits, dim = 1)
        return logits#, probas

# Multilayer Perception
class MLP(torch.nn.Module):

    def __init__(self, num_features, num_hidden, num_classes):
        super(MLP, self).__init__()

        self.num_classes = num_classes

        # The First Hidden Layer
        self.linear_1 = torch.nn.Linear(num_features, num_hidden)
        self.linear_1.weight.detach().normal_(0.0, 0.1) # mean, std
        self.linear_1.bias.detach().zero_()

        # Output Layer
        self.linear_out = torch.nn.Linear(num_hidden, num_classes)
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()

    def forward(self, x):
        out = self.linear_1(x)
        out = torch.sigmoid(out)
        logits = self.linear_out(out)
        return logits

def compute_accuracy(net, data_loader):
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(-1, 28 * 28).to(device)
            targets = targets.to(device)
            logits = net.forward(features)
            predicted_labels = torch.argmax(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
            accuracy = correct_pred.float() / num_examples
        return accuracy

def compute_loss(net, data_loader):
    curr_loss = 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(-1, 28 * 28).to(device)
            targets = targets.to(device)
            logits = net.forward(features)
            loss = F.cross_entropy(logits, targets)

        curr_loss += loss
        return float(curr_loss)

def calculating(num_epochs):
    # start_time = time.time()
    train_accuracy_list = []
    test_accuracy_list = []
    train_loss_list = []
    test_loss_list = []
    train_dataset, test_dataset, train_loader, test_loader = get_dataset(batch_size)
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.view(-1, 28 * 28).to(device)
            targets = targets.to(device)

            # Forward and Back Prop
            logits = model(features)

            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # Update Model Parameters
            optimizer.step()

            # Logging
            # if not batch_idx % 50:
            #     print ('Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f'
            #            % (epoch + 1, num_epochs, batch_idx, len(train_loader), loss.item()))

        train_accuracy = compute_accuracy(model, train_loader)
        train_accuracy_list.append(train_accuracy)
        test_accuracy = compute_accuracy(model, test_loader)
        test_accuracy_list.append(test_accuracy)
        train_loss = compute_loss(model, train_loader)
        train_loss_list.append(train_loss)
        test_loss = compute_loss(model, test_loader)
        test_loss_list.append(test_loss)
        # print('Training Time: %.2f min' % ((time.time() - start_time) / 60))
        print("[Epoch]: %i, [Train Acc]: %.4f, [Train Loss]: %.4f, [Test Acc]: %.4f, [Test Loss]: %.4f"
              % (epoch + 1, train_accuracy, train_loss, test_accuracy, test_loss))

    return train_accuracy_list, test_accuracy_list, train_loss_list, test_loss_list

if __name__ == "__main__":
    # Hyperparameters
    random_seed = 1
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 8

    # Architecture
    num_features = 28 * 28
    num_hidden = 100
    num_classes = 10

    # Manual seed for deterministic data loader
    torch.manual_seed(random_seed)

    # Three Models
    model = SoftmaxRegression(num_features = num_features, num_classes = num_classes)
    # model = MLP(num_features = num_features, num_hidden = num_hidden, num_classes = num_classes)
    model = model.to(device)

    # Cost and Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    train_accuracy_list, test_accuracy_list, train_loss_list, test_loss_list = calculating(num_epochs)
    epochs = [i for i in range(1, num_epochs + 1)]

    # Training & Testing Accuracy Plot
    plt.plot(epochs, train_accuracy_list, color = 'tomato', label = 'Train Accuracy')
    plt.plot(epochs, test_accuracy_list, color = 'limegreen', label = 'Test Accuracy')
    plt.title('Train and Test Accuracy of MNIST Dataset (Soft-max)')
    # plt.title('Train and Test Accuracy of MNIST Dataset (MLP)')
    plt.xticks(range(1, num_epochs + 1, 1))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc = "lower right")
    plt.show()

    # Training & Testing Loss Plot
    plt.plot(epochs, train_loss_list, color = 'tomato', label = 'Train Loss')
    plt.plot(epochs, test_loss_list, color = 'limegreen', label = 'Test Loss')
    plt.title('Train and Test Loss of MNIST Dataset (Soft-max)')
    # plt.title('Train and Test Loss of MNIST Dataset (MLP)')
    plt.xticks(range(1, num_epochs + 1, 1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc = "upper right")
    plt.show()

    # Show MNIST picture
    # train_dataset, test_dataset, train_loader, test_loader = get_dataset(batch_size)
    # for features, targets in test_loader:
    #     break
    # fig, ax = plt.subplots(1, 4)
    # for i in range(4):
    #     ax[i].imshow(features[i].view(28, 28), cmap = matplotlib.cm.binary)
    # plt.show()

    # Predictions
    # _, predictions = model.forward(features[: 4].view(-1, 28*28).to(device))
    # predictions = torch.argmax(predictions, dim = 1)
    # print('Predicted Labels', predictions)
    # model.forward(features[:4].view(-1, 28*28).to(device))

