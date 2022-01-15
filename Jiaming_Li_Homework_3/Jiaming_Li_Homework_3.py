# Jiaming Li
# University of Ottawa
# CSI 5340 Intro to Deep Learning and Reinforcement Learning
# Homework 3

import torch
from torchtext.legacy import data, datasets
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def get_dataset(batch_size):
    text = data.Field(tokenize = 'spacy', lower = True, include_lengths = True, tokenizer_language='en_core_web_md')
    label = data.LabelField(dtype = torch.float)
    train_data, test_data = datasets.IMDB.splits(text, label)

    # print('Number of training examples: ', len(train_data.examples))
    # print('Number of testing examples: ', len(test_data.examples))
    # print(vars(train_data.examples[0]))

    text.build_vocab(train_data, max_size = 10000, min_freq = 5, vectors = "glove.6B.50d")
    label.build_vocab(train_data, min_freq = 5)

    # print(vars(label.vocab))
    # print('Unique tokens in text vocabulary: ', len(text.vocab))
    # print('Unique tokens in label vocabulary: ', len(label.vocab))

    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data),
        batch_size = batch_size,
        device = device
    )

    return text, label, train_iterator, test_iterator

# LSTM Model
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = True, dropout = dropout)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted = False)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        output = self.linear(hidden)
        return output

#Vanilla RNN Model
class VanillaRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = True, dropout = dropout)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded)
        hidden_output = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim = 1))
        output = self.linear(hidden_output)
        return output

def calculation(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct = (rounded_preds == batch.label).float()
            acc = correct.sum() / len(correct)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_acc / len(iterator), epoch_loss / len(iterator)

# Model training function
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)

        rounded_preds = torch.round(torch.sigmoid(predictions))
        correct = (rounded_preds == batch.label).float()
        acc = correct.sum() / len(correct)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    # for i, batch in enumerate(iterator):
    #     text = batch.text
    #     text = pad_sequence(torch.FloatTensor([item.detach().numpy() for item in text]))
    #     label = batch.label
    #     # label = torch.FloatTensor(label)
    #     optimizer.zero_grad()
    #     output = model(text)
    #
    #     epoch_acc += torch.sum(torch.eq(output.argmax(1), label))
    #     total_count += len(label)
    #
    #     label = label.unsqueeze(1)
    #     loss = criterion(output, label.float())
    #
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    #     optimizer.step()
    #     epoch_loss += loss.item()

    return epoch_acc / len(iterator), epoch_loss / len(iterator)

if __name__ == "__main__":

    text, label, train_iterator, test_iterator = get_dataset(batch_size = 50)
    input_dim = len(text.vocab)
    # output_dim = len(label.vocab)
    output_dim = 1
    emb_dim = 50
    hidden_dim = 500
    layers = 2
    dropout = 0.5
    epochs = 5
    pad_idx = text.vocab.stoi[text.pad_token]

    # initializing our model
    model = LSTM(input_dim, emb_dim, hidden_dim, output_dim, layers, dropout, pad_idx)
    # model = VanillaRNN(input_dim, emb_dim, hidden_dim, output_dim, layers, dropout)
    model = model.to(device)

    # loading pretrained word embedding
    model.embedding.weight.data.copy_(text.vocab.vectors)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.BCEWithLogitsLoss().to(device)

    train_accuracy_list = []
    test_accuracy_list = []
    train_loss_list = []
    test_loss_list = []
    epoch_list = [i for i in range(1, epochs + 1)]

    for epoch in range(epochs):
        train_acc, train_loss = train(model, train_iterator, optimizer, criterion)
        test_acc, test_loss = calculation(model, test_iterator, criterion)

        train_accuracy_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_accuracy_list.append(test_acc)
        test_loss_list.append(test_loss)

        print("[Epoch]: %i, [Train Acc]: %.4f, [Train Loss]: %.4f, [Test Acc]: %.4f, [Test Loss]: %.4f"
              % (epoch + 1, train_acc, train_loss, test_acc, test_loss))

    print('train_accuracy_list = ', train_accuracy_list)
    print('train_loss_list = ', train_loss_list)
    print('test_accuracy_list = ', test_accuracy_list)
    print('test_loss_list = ', test_loss_list)

    # Training & Testing Accuracy Plot
    plt.plot(epoch_list, train_accuracy_list, color='tomato', label = 'Train Accuracy')
    plt.plot(epoch_list, test_accuracy_list, color='limegreen', label = 'Test Accuracy')
    plt.legend(loc='upper right')
    plt.xticks(range(1, epochs + 1, 1))
    plt.yticks(np.arange(0, 1.1, step = 0.1))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy of IMDB Dataset (Hidden Dimension 500)')
    plt.show()

    # Training & Testing Loss Plot
    plt.plot(epoch_list, train_loss_list, color='tomato', label = 'Train Loss')
    plt.plot(epoch_list, test_loss_list, color='limegreen', label = 'Test Loss')
    plt.legend(loc='upper right')
    plt.xticks(range(1, epochs + 1, 1))
    plt.yticks(np.arange(0, 1.1, step = 0.1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss of IMDB Dataset (Hidden Dimension 500)')
    plt.show()



