# -*- coding: utf-8 -*-

"""
This code simulate the pendulum system using
scipy.integrate.odeint package
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def pendulum(var_x, unused_t, grav, damping_constant, torque):
    """
    pendulum system vector-space function
    """
    var_x1, var_x2 = var_x
    dxdt = [var_x2, \
    -grav*np.sin(var_x1) - damping_constant*var_x2 + torque]
    return dxdt

# inital condition
G = 9.8  # gravitational constant

# damping constant alpha collection of two different cases
ALPHA_COLLECTION = [0.3, 0.7]
T = 0  # the control input

# inital theta collection of two different cases
X1_0_COLLECTION = [np.pi*3/4, np.pi/4]
X2_0 = 0  # inital omega

# simulation setup
SIM_TIME = np.linspace(0, 9.9, 400)
# y = []  # the output collection of four cases

plt.subplots(2, 2, sharex='all', sharey='all', figsize=(14, 8))
# plt.figure()

# four cases
for i in range(4):
    # choose x1_0 with rem,
    # when i = 0 or 2, x1_0 is in the first case,
    # when i = 1 or 3, in another one
    x0 = [X1_0_COLLECTION[i%2], X2_0]

    # choose alpha with mod,
    # when i = 0 or 1, alpha is in the first case,
    # when i = 2 or 3, in another one
    alpha = ALPHA_COLLECTION[i//2]

    # solve
    y = odeint(pendulum, x0, SIM_TIME, args=(G, alpha, T))

    # plot
    plt.subplot(2, 2, i+1)
    plt.plot(SIM_TIME, y[:, 0], label='x1:theta')
    plt.plot(SIM_TIME, y[:, 1], label='x2:omega')
    plt.title('x1_0={:.2f},x2_0={:.2f},alpha={:.2f},T={:.2f}'\
            .format(x0[0], x0[1], alpha, T))
    plt.legend(loc='best')
    plt.ylim(-6, 6)
    if i >= 2:
        plt.xlabel('time')
    plt.grid()

# save and show
plt.savefig(r'./HW1/img/pendulum_sim.png')
plt.show()
