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


"""
function: growth_rates(Gammas, **kwargs)

Plots the relative gammas of all the agents in the system, with their respective error.

Inputs:
Gammas: Relative Growth Rate of the agents.
"""
def growth_rates(Gammas, **kwargs):
    import numpy as np
    if "Gammas_def" in kwargs: Gammas = np.copy(Gammas - kwargs["Gammas_def"])

    N = len(Gammas)
    Gamma_mean = np.mean(Gammas, axis = 1) * 100
    Gamma_std = np.std(Gammas, axis = 1) * 100
    Gamma_error = Gamma_std/np.sqrt(N)

    bincenters = range(N)
    plt.bar(bincenters, Gamma_mean, yerr=Gamma_error)
    plt.xlabel(r"Agents")
    plt.ylabel(r"Relative long term growth rate")
    plt.show()
    return


"""
function: graph(A)

Plots Graph of network

Inputs:
A: Adjecency matrix of the graph
"""
def graph(A):
    import networkx as nx
    G = nx.from_numpy_array(A)
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=700, edge_color='gray', font_weight='bold')
    plt.show()


if __name__ == '__main__':
    import numpy as np
    X_coop = np.loadtxt('data/X_coop.txt')
    X_def = np.loadtxt('data/X_def.txt')
    evolution(X_coop)
    evolution(X_def)
