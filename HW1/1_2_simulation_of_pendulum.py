import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def pendulum(x, t, g, alpha, T):
    '''
    pendulum system vector-space function
    '''
    x1, x2 = x
    dxdt = [x2, -g*np.sin(x1) - alpha*x2 + T ]
    return dxdt

##  inital condition
g = 9.8  # gravitational constant

# damping constant alpha collection of two different cases
alpha_c = [0.3, 0.7]
T = 0  # the control input

# inital theta collection of two different cases
x1_0_c = [np.pi*3/4, np.pi/4]
x2_0 = 0  # inital omega

## simulation setup
t = np.linspace(0, 9.9, 400)
# y = []  # the output collection of four cases

plt.subplots(2, 2, sharex='all', sharey='all', figsize=(14, 8))
# plt.figure()

for i in range(4):
    '''
    four cases
    choose x1_0 with rem:
        when i = 0 or 2, x1_0 is in the first case; 
        when i = 1 or 3, in another one
    choose alpha with mod:
        when i = 0 or 1, alpha is in the first case; 
        when i = 2 or 3, in another one
    '''
    x0 = [x1_0_c[i%2], x2_0]
    alpha = alpha_c[i//2]

    ## solve
    y = odeint(pendulum, x0, t, args=(g, alpha, T))

    ## plot 
    plt.subplot(2, 2, i+1)
    plt.plot(t, y[:, 0], label='x1:theta')
    plt.plot(t, y[:, 1], label='x2:omega')
    plt.title('x1_0={:.2f},x2_0={:.2f},alpha={:.2f},T={:.2f}'\
            .format(x0[0], x0[1], alpha, T))
    plt.legend(loc='best')
    plt.ylim(-6, 6)
    if(i>=2):plt.xlabel('time')
    plt.grid()

## save and show
plt.savefig(r'./HW1/img/pendulumSim.png')
plt.show()