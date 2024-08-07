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



