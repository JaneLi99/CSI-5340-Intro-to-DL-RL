# Jiaming Li
# University of Ottawa
# CSI 5340 Intro to Deep Learning and Reinforcement Learning
# Homework 2 Question 2

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.special import softmax

def proof(K, m):

    x = np.random.normal(size = m)
    W = np.random.normal(size=(K, m))
    A = np.random.normal(size=(K, K))
    B = np.random.normal(size=(K, K))
    C = np.random.normal(size=(K, m))

    # Two Models
    H1 = softmax(np.matmul(W, x), axis = 0)
    H2 = softmax(np.matmul(np.matmul(A + B, C), x), axis = 0)

    return np.mean(H1), np.var(H1), np.mean(H2), np.var(H2)

if __name__ == "__main__":
    H1_mean_list = []
    H1_var_list = []
    H2_mean_list = []
    H2_var_list = []
    n = 10
    for i in range(n):
        m = random.randint(10, 1000)
        K = random.randint(10, 1000)
        H1_mean, H1_var, H2_mean, H2_var = proof(K, m)
        H1_mean_list.append(H1_mean)
        H1_var_list.append(H1_var)
        H2_mean_list.append(H2_mean)
        H2_var_list.append(H2_var)
        # print('m = ', m, ', K = ', K)
    # print('H1 Mean List', H1_mean_list)
    # print('H1 Variance List', H1_var_list)
    # print('H2 Mean List', H2_mean_list)
    # print('H2 Variance List', H2_var_list)

    x_axis = [i for i in range(1, n + 1)]
    # Mean Value Plot
    # plt.plot(x_axis, H1_mean_list, color = 'tomato', label = 'H1 Mean')
    # plt.plot(x_axis, H2_mean_list, color = 'limegreen', label = 'H2 Mean')
    # plt.title('Mean Value of H1 and H2')
    # plt.xticks(range(1, n + 1, 1))
    # plt.xlabel('Parameters')
    # plt.ylabel('Mean Value')
    # plt.legend(loc = "upper right")
    # plt.show()

    # Variance Plot
    plt.plot(x_axis, H1_mean_list, color = 'tomato', label = 'H1 Variance')
    plt.plot(x_axis, H2_mean_list, color = 'limegreen', label = 'H2 Variance')
    plt.title('Variance Value of H1 and H2')
    plt.xticks(range(1, n + 1, 1))
    plt.xlabel('Parameters')
    plt.ylabel('Variance Value')
    plt.legend(loc = "upper right")
    plt.show()

