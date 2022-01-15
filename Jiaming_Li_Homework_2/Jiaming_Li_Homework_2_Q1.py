# Jiaming Li
# University of Ottawa
# CSI 5340 Intro to Deep Learning and Reinforcement Learning
# Homework 2 Question 1

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(y):
    u = 1.0 / (1.0 + np.exp(y))
    return u

def back_propagation(k, epochs_list, learning_rate):
    A =  np.random.rand(k, k)
    B =  np.random.rand(k, k)
    x = np.ones(k)
    Loss_list = []
    for epoch in range(len(epochs_list)):
        y = np.matmul(A, x)
        u = sigmoid(y)
        v = np.matmul(B, x)
        z = np.matmul(A, np.multiply(u, v))
        w = np.matmul(A, z)

        Loss = np.sum(np.multiply(w, w))
        Loss_list.append(Loss)

        z_grad = np.transpose(A)
        y_grad = sigmoid(y) * (1 - sigmoid(y))
        A_grad = 2 * np.matmul(w, z) + \
                 np.matmul(2 * np.matmul(z_grad, w), np.multiply(u, v)) + \
                 np.matmul(np.multiply(sigmoid(y_grad), np.ones(k) - sigmoid(y_grad)), x)
        A = A - learning_rate * A_grad
        B_grad = np.matmul(np.multiply(np.matmul(2 * np.matmul(z_grad, z_grad), w), u), x)
        B = B - learning_rate * B_grad

    print('A: ', A)
    print('B: ', B)
    return Loss_list

if __name__ == "__main__":
    k = 10
    learning_rate = 0.001
    epochs_list = [epoch for epoch in range(1000)]
    Loss_list = back_propagation(k, epochs_list, learning_rate)
    #print('loss list: ', Loss_list)

    plt.plot(epochs_list, Loss_list, color = 'steelblue')
    plt.title('Value of Loss Function (Epochs = 1000, k = 10)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.show()

