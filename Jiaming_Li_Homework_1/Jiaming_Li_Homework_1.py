# Jiaming Li
# University of Ottawa
# CSI 5340 Intro to Deep Learning and Reinforcement Learning
# Homework 1

import numpy as np
import matplotlib.pyplot as plt
import math

def getData(N, sigma):
    X = np.random.uniform(0, 1, N)
    Z = np.random.normal(0, sigma, N)
    Y = np.cos(2 * math.pi * X) + Z
    # plt.scatter(X, Y, color='steelblue', marker= '.', s = 20)
    # plt.title("getData")
    # plt.show()
    data_sample = np.vstack((X, Y)).T
    return data_sample

def getMSE(Y, Y_estimated):
    difference = Y - Y_estimated
    squared_difference = difference ** 2
    MSE = np.mean(squared_difference)
    return MSE

def fitData(data, d):
    epochs = 1000
    learning_rate = 0.01
    N = len(data)
    Y = data[:, 1].reshape((N, 1))
    x_update = np.ones((1, N))
    for i in range(d):
        temp = data[:, 0] ** (i + 1)
        x_update = np.vstack((x_update, temp))
    X = x_update.T

    # Without regularization
    # theta = np.random.random([d + 1, 1])
    # J = np.zeros(epochs)
    # for iter in range(epochs):
    #     theta = theta - (1 / N) * learning_rate* ((X.dot(theta) - Y).T.dot(X).T)
    #     J[iter] = getMSE(X.dot(theta), Y)
    #     E_in = J[epochs - 1]
    # return theta, E_in

    # With regularization (weight decay)
    weight_theta = np.random.random([d + 1, 1])
    lambda_value = 1
    New_MSE_list = []
    J = np.zeros(epochs)
    for iter in range(epochs):
        regularization_term = (lambda_value / 2 * N) * np.sum(np.square(weight_theta))
        New_MSE = getMSE(X.dot(weight_theta), Y) + regularization_term
        gradient = (1 / N) * ((X.dot(weight_theta) - Y).T.dot(X).T + (lambda_value * weight_theta))
        weight_theta = weight_theta - learning_rate * gradient
        J[iter] = getMSE(X.dot(weight_theta), Y)
        E_in = J[epochs - 1]
        New_MSE_list.append(New_MSE)
        print('The epoch is: ', iter, '. The error (MSE) is: ', New_MSE_list)
    return weight_theta, E_in

def experiment(N, d, sigma):
    M = 50
    E_in_list = []
    E_out_list = []
    theta_list = []
    for m in range(M):
        data_traning = getData(N, sigma)
        theta, E_in = fitData(data_traning, d)
        data_test = getData(2000, sigma)
        E_out_x = np.ones((1, len(data_test)))
        E_out_y = data_test[:, 1].reshape((len(data_test), 1))
        for i in range(d):
            temp = data_test[:, 0] ** (i + 1)
            E_out_x = np.vstack((E_out_x, temp))
        E_out = getMSE(E_out_x.T.dot(theta), E_out_y)
        theta_list.append(theta)
        E_in_list.append(E_in)
        E_out_list.append(E_out)

    E_in_average = np.mean(E_in_list)
    E_out_average = np.mean(E_out_list)
    theta_average=np.mean(theta_list, axis=0)

    data_bias= getData(2000, sigma)
    E_bias_x = np.ones((1, len(data_bias)))
    E_bias_y = data_bias[:, 1].reshape((len(data_bias), 1))
    for i in range(d):
        temp = data_bias[:, 0] ** (i + 1)
        E_bias_x = np.vstack((E_bias_x, temp))
    E_bias = getMSE(E_bias_x.T.dot(theta_average), E_bias_y)
    return E_in_average, E_out_average, E_bias

if __name__ == "__main__":
    N_list = [2, 5, 10, 20, 50, 100, 200]
    d_list = [d for d in range(20)]
    sigma_list = [0.01, 0.1, 1]

    E_in_y = []
    E_out_y = []
    E_bias_y = []

    # when N is different
    for N in range(len(N_list)):
        E_in_average, E_out_average, E_bias = experiment(N_list[N], 10, 0.1)
        E_in_y.append(E_in_average)
        E_out_y.append(E_out_average)
        E_bias_y.append(E_bias)
    plt.plot(N_list, E_in_y, 'tomato', label = 'E_in')
    plt.plot(N_list, E_out_y, 'steelblue', label = 'E_out')
    plt.plot(N_list, E_bias_y, 'limegreen', label = 'E_bias')
    plt.title('E_in, E_out, E_bias of Different Sample Size (with regularization)')
    plt.xlabel('Sample Size N')
    plt.ylabel('MSE Value')
    plt.legend(loc = "upper right")
    plt.show()

    # # when d is different
    # for d in range(len(d_list)):
    #     E_in_average, E_out_average, E_bias = experiment(100, d_list[d], 0.1)
    #     E_in_y.append(E_in_average)
    #     E_out_y.append(E_out_average)
    #     E_bias_y.append(E_bias)
    # plt.plot(d_list, E_in_y, 'tomato',label = 'E_in')
    # plt.plot(d_list, E_out_y, 'steelblue',label = 'E_out')
    # plt.plot(d_list, E_bias_y, 'limegreen',label = 'E_bias')
    # plt.title('E_in, E_out, E_bias of Different Model Complexity (without regularization)')
    # plt.xlabel('Model Complexity d')
    # plt.ylabel('MSE Value')
    # my_x_ticks = np.arange(0, 20, 2)
    # plt.xticks(my_x_ticks)
    # plt.legend()
    # plt.show()

    # # when sigma is different
    # for sigma in range(len(sigma_list)):
    #     E_in_average, E_out_average, E_bias = experiment(100, 10, sigma_list[sigma])
    #     E_in_y.append(E_in_average)
    #     E_out_y.append(E_out_average)
    #     E_bias_y.append(E_bias)
    # plt.plot(sigma_list, E_in_y, 'tomato', label = 'Ein')
    # plt.plot(sigma_list, E_out_y, 'steelblue', label = 'Eout')
    # plt.plot(sigma_list, E_bias_y, 'limegreen', label = 'Ebias')
    # plt.title('E_in, E_out, E_bias of Different Noise Level (without regularization)')
    # plt.xlabel('Noise Level Sigma')
    # plt.ylabel('MSE Value')
    # plt.legend()
    # plt.show()

