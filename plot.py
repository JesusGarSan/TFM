"""
This module contains functions to plot the evolution of the agent models.
It serves an easy and convenient way to quickly plot the results of evolution
"""
import matplotlib.pyplot as plt

"""
function: evolution

Plots the evolution over time of an agent model

Inputs:
x: Values of the Agents over time
legend: Boolean. Determines if the legend will be shown or not
"""
def evolution(x, legend = True):
    import matplotlib.pyplot as plt
    steps, N = x.shape
    time = range(steps)
    for n in range(N):
        plt.scatter(time, x[:, n], label=f"Agent {n+1}", s = 1)
        plt.plot(time, x[:,n])

    plt.xlabel("Time steps")
    plt.ylabel("Agent values")
    if legend: plt.legend()
    if not legend: plt.legend('',frameon=False)
    plt.show()


if __name__=="__main__":
    import numpy as np
    N = 2
    x0 = 1
    a_i = 0.2
    x_ini = np.ones(N)*x0
    a = np.ones(N) * a_i
    steps = int(1e3)
    mu = 1
    sigma = 0.1  

    from evolution import *

    X, _ = evolve_cooperation(x_ini, a, mu, sigma,)
    