# -*- coding: utf-8 -*-

"""
This code simulate the weight trajectory
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# initialize the condition
A = np.array([[1, 1], [1, 1]])
# case 1
Q = np.array([[5, 3], [3, 2]])
b = np.array([1, -1])
# case 2
# Q = np.array([[1, 2], [3, 4]])
# b = np.array([1, -1])
x0 = np.array([1, 1, 0, 0])

# simulation time
SIM_LEN = 40

def accelerated_gradient(t, x):
    """
    accelerated gradient algorithm system
    """
    x_tmp = -np.matmul(Q+Q.T, [x[0], x[1]])-np.matmul(A, [x[2], x[3]])-b
    return np.array([x[2], x[3], x_tmp[0], x_tmp[1]])

def compute_loss(weights):
    """compute the loss, given a series of weight"""
    computed_loss = []
    for i in range(weights.shape[1]):
        w = np.array(weights[0:2, i])
        computed_loss.append(np.matmul(w.T, np.matmul(Q, w))+np.matmul(b.T, w)) 
    return computed_loss

sol = solve_ivp(accelerated_gradient, [0, SIM_LEN], x0, method='LSODA', dense_output=True)

t = np.linspace(0, SIM_LEN, SIM_LEN*30)
z = sol.sol(t)
loss = compute_loss(z)
plt.plot(t, loss)
plt.plot(t, z[0:2, :].T)
plt.xlabel('t')
plt.legend(['loss', 'w1', 'w2'], shadow=True)
plt.title('Gradient Algorithm System')
plt.show()
