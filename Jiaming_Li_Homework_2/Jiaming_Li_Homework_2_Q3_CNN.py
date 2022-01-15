# Jiaming Li
# University of Ottawa
# CSI 5340 Intro to Deep Learning and Reinforcement Learning
# Homework 2 Question 3
# Convolutional Neural Networks

import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dataset(batch_size):
    # MNIST Dataset
    # Train and Test data loaders
    train_dataset = datasets.MNIST(root = 'D:/李佳明/University_of_Ottawa/Fall_2021/CSI_5340/Jiaming_Li_Homework_2',
                                   train = True,
                                   transform = transforms.ToTensor(),
                                   download = True)
    test_dataset = datasets.MNIST(root = 'D:/李佳明/University_of_Ottawa/Fall_2021/CSI_5340/Jiaming_Li_Homework_2',
                              train = False,
                              transform = transforms.ToTensor())

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    return train_loader, test_loader

# CNN Model
class CNN(nn.Module):
    def __init__(self, dropout_p, num_hidden_units, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.dropout_conv2 = nn.Dropout2d(dropout_p)
        self.fc1 = nn.Linear(320, num_hidden_units)
        self.fc2 = nn.Linear(num_hidden_units, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout_conv2(self.conv2(x)), 2))
        x = x.view(-1, 320) # Flatten
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return x # logits

def process_batch(inputs, targets, model, criterion, optimizer, is_training):
    if is_training:
        X_batch = Variable(inputs, requires_grad = False)
    else:
        with torch.no_grad():
            X_batch = Variable(inputs, requires_grad = False)
    y_batch = Variable(targets.long(), requires_grad = False)

    # Forward pass
    scores = model(X_batch) #logits

    # Loss
    loss = criterion(scores, y_batch)

    # Accuracu
    scores, predicted = torch.max(scores, 1)
    accuracy = (y_batch.data == predicted.data).sum() / float(len(y_batch))

    if is_training:

        # Use autograd to do backprop. This will compute the gradient w.r.t loss for all Varaiable that have
        # requies_grad = True. So,our w1 nd w2 will now have gradient components we can access
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss, accuracy

def train(model, criterion, optimizer, train_loader, test_loader, num_epochs, log_interval, learning_rate, decay_rate):

    # Metrics
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # Training
    for num_train_epoch in range(num_epochs):

        # Timer
        # start = time.time()

        # Decay learning rate
        learning_rate = learning_rate * (decay_rate ** (num_train_epoch // 1.0))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        #Metrics
        train_batch_loss = 0.0
        train_batch_accuracy = 0.0

        for train_batch_num, (inputs, target) in enumerate(train_loader):
            # Get Metrics
            model.train()
            loss, accuracy = process_batch(inputs, target, model, criterion, optimizer, model.training)

            # Add to batch scalars
            train_batch_loss += loss.item() / float(len(inputs))
            train_batch_accuracy += accuracy

        # Add to global metrics
        train_loss.append(train_batch_loss / float(train_batch_num + 1))
        train_acc.append(train_batch_accuracy / float(train_batch_num + 1))

        # Testing
        model.eval()
        for num_test_epoch in range(1):

            #Metrics
            test_batch_loss = 0.0
            test_batch_accuracy = 0.0

            for test_batch_num, (inputs, target) in enumerate(test_loader):
                # Get Metrics
                model.eval()
                loss, accuracy = process_batch(inputs, target, model, criterion, optimizer, model.training)

                # Add to batch scalars
                test_batch_loss += loss.item() / float(len(inputs))
                test_batch_accuracy += accuracy

            # Add to global metrics
            test_loss.append(test_batch_loss / float(test_batch_num + 1))
            test_acc.append(test_batch_accuracy / float(test_batch_num + 1))

            verbose_condition = (num_train_epoch == 0) or (num_train_epoch % log_interval == 0) or (num_train_epoch == num_epochs - 1)

            # Verbose
            if verbose_condition:
                # time_remain = (time.time() - start) * (num_epochs - (num_train_epoch + 1))
                # minutes = time_remain / 60
                # print("Time remaining: %im" % (minutes))
                print("[Epoch]: %i, [Train Acc]: %.4f, [Train Loss]: %.4f, [Test Acc]: %.4f, [Test Loss]: %.4f"
                      % (num_train_epoch + 1, train_batch_accuracy / float(train_batch_num + 1),
                         train_batch_loss / float(train_batch_num + 1),  test_batch_accuracy / float(test_batch_num + 1),
                         test_batch_loss / float(test_batch_num + 1)))

    return train_acc,  train_loss, test_acc, test_loss

if __name__ == "__main__":
    # Hyperparameters
    random_seed = 1
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001
    dropout_p = 0.5
    log_interval = 1 # epochs
    num_hidden_units = 50
    num_classes = 10 #MNIST
    decay_rate = 0.9999

    torch.manual_seed(random_seed)

    train_loader, test_loader = get_dataset(batch_size)

    # Initialize model components
    model = CNN(dropout_p, num_hidden_units, num_classes)

    # Cost and Optimizer
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    train_accuracy_list,  train_loss_list, test_accuracy_list, test_loss_list = \
        train(model, criterion, optimizer, train_loader, test_loader, num_epochs, log_interval, learning_rate, decay_rate)

    epochs = [i for i in range(1, num_epochs + 1)]

    # Training & Testing Accuracy Plot
    plt.plot(epochs, train_accuracy_list, color = 'tomato', label = 'Train Accuracy')
    plt.plot(epochs, test_accuracy_list, color = 'limegreen', label = 'Test Accuracy')
    plt.title('Train and Test Accuracy of MNIST Dataset (CNN)')
    plt.xticks(range(1, num_epochs + 1, 1))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc = "lower right")
    plt.show()

    # Training & Testing Loss Plot
    plt.plot(epochs, train_loss_list, color = 'tomato', label = 'Train Loss')
    plt.plot(epochs, test_loss_list, color = 'limegreen', label = 'Test Loss')
    plt.title('Train and Test Loss of MNIST Dataset (CNN)')
    plt.xticks(range(1, num_epochs + 1, 1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc = "upper right")
    plt.show()






