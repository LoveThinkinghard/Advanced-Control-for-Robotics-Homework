# -*- coding: utf-8 -*-

"""
This code simulate the gradient algorithm system
using scipy.integrate.solve_ivp,
which is recommonded by the official
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# initialize the condition
A = np.array([[1, 1], [1, 1]])
# choose a case
CASE = 2
if CASE == 1:
    # case 1
    Q = np.array([[5, 3], [3, 2]])
    B = np.array([1, -1])
elif CASE == 2:
    # case 2
    Q = np.array([[1, 2], [3, 4]])
    B = np.array([1, -1])
else:
    raise Exception("Don't exist case {}, else \
        choose between 1 and 2".format(CASE))
X0 = np.array([1, 1, 0, 0])

# simulation time
SIM_LEN = 50
def accelerated_gradient(unused_t, var_x):
    """
    accelerated gradient algorithm system.
    Args:
        unused_t: used by solver.
        var_x: the input x should be one-dimension.
    Returns:
        An array which is the next epoch var_x, so having the same shape.
    """
    x_tmp = -np.matmul(Q+Q.T, [var_x[0], var_x[1]])-\
        np.matmul(A, [var_x[2], var_x[3]])-B
    return np.array([var_x[2], var_x[3], x_tmp[0], x_tmp[1]])

def compute_loss(weights):
    """
    compute the loss, given a series of weight
    Args:
        weights: shape(4, n), n is the number of time points
    Returns:
        A list who has the length of n.
    """
    computed_loss = []
    for i in range(weights.shape[1]):
        weight = np.array(weights[0:2, i])
        computed_loss.append(np.matmul(weight.T, \
            np.matmul(Q, weight))+np.matmul(B.T, weight))
    return computed_loss

SOLUTION = solve_ivp(accelerated_gradient, [0, SIM_LEN], X0, \
    method='LSODA', dense_output=True)

TIME_SERIES = np.linspace(0, SIM_LEN, SIM_LEN*30)
WEIGHTS = SOLUTION.sol(TIME_SERIES)
LOSS = compute_loss(WEIGHTS)

plt.subplot(2, 1, 1)
plt.plot(TIME_SERIES, LOSS)
plt.legend(['loss'])
plt.grid()
plt.title('Accelerated Gradient Algorithm System Case {}'.format(CASE))

plt.subplot(2, 1, 2)
plt.plot(TIME_SERIES, WEIGHTS[0:2, :].T)
plt.legend(['w1', 'w2'])
plt.grid()
plt.xlabel('time')

plt.savefig(r'./HW1/img/accelerated_gradient_simulation_case_{}.png'\
    .format(CASE))
plt.show()
