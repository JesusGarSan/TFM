"""
This module contains all the functions needed to evolve the agent models.
It takes charge of all the underlaying mathematics
"""

import numpy as np

"""
function: _evolve_cooperation(x, a, mu, sigma, steps = 1000)

Evolution of the Stochastic Multiplicative Growth process
This evolution function considers that all agents play for the entire evolution period.

Inputs:
x: Initial values of the agents
a: Sharing parameters of each agent
mu: mean of the normal distribution used for the stochasticity
sigma: Standard deviation of the normal distribution used for the stochasticity
steps: Number of time steps to consider.

Returns:
X: [steps, len(x)] array. Value of the agents along the evolution
"""
def _evolve_cooperation(x, a, mu, sigma, steps = 1000):
    assert len(x) == len(a), "Número de agentes inconsistente \n Inconsistent number of agents"
    N = len(x)
    time = range(1, steps+1) 
    x_ini = np.copy(x)

    X = np.zeros((steps+1, N))
    X[0] = x_ini

    for i in time:
        dseta = np.random.normal(mu, sigma, N)
        x = x * dseta*(1 - a) + np.mean(a*x*dseta)
        X[i] = x
            
    return X

"""
function: _evolve_defection(x, a, mu, sigma, steps = 1000)

Evolution of the Stochastic Multiplicative Growth process compared to the defective case.
This evolution function considers that all agents play for the entire evolution period.
It will evolve the system according to the sharing parameters as well as complete
defection using the same random numbers. This way we can compare the evolution in 
full defection mode compared to a cooperative case.

Inputs:
x: Initial values of the agents
a: Sharing parameters of each agent
mu: mean of the normal distribution used for the stochasticity
sigma: Standard deviation of the normal distribution used for the stochasticity
steps: Number of time steps to consider.

Returns:
X_coop: [steps, len(x)] array. Value of the agents along the cooperative evolution
X_def: [steps, len(x)] array. Value of the agents along the defective evolution
"""
def _evolve_defection(x, a, mu, sigma, steps = 1000):
    assert len(x) == len(a), "Número de agentes inconsistente \n Inconsistent number of agents"
    N = len(x)

    time = range(1, steps+1) 
    x_def = np.copy(x)
    x_ini = np.copy(x)

    X_coop = np.zeros((steps+1, N))
    X_def = np.zeros((steps+1, N))
    X_coop[0], X_def[0] = x_ini, x_ini
    for i in time:
        dseta = np.random.normal(mu, sigma, N)
        x = x * dseta*(1 - a) + np.mean(a*x*dseta)
        x_def = x_def*dseta

        X_coop[i] = x
        X_def[i] = x_def
            
    return X_coop, X_def

"""
function: evolve_network(Adj, x, a, mu, sigma, steps = 1000)

Evolution of the Stochastic Multiplicative Growth process in a network.
This evolution function considers that all agents play for the entire evolution period.
The sharing parameter of an agent is applied to every one of its links, so that the total amount
of value it shares is its share parameter times its number of links.

Inputs:
Adj: Adjecency Matrix.
x: Initial values of the agents
a: Sharing parameters of each agent
mu: mean of the normal distribution used for the stochasticity
sigma: Standard deviation of the normal distribution used for the stochasticity
steps: Number of time steps to consider.

Returns:
X: [steps, len(x)] array. Value of the agents along the evolution
"""
def _evolve_network_cooperation(Adj, x, a, mu, sigma, steps = 1000):
    assert len(x) == len(a), "Número de agentes inconsistente \n Inconsistent number of agents"
    N = len(x)
    assert (N,N) == Adj.shape, "Dimensiones de la matriz de adyacencia inconsistentes con el número de agentes \n The dimensions of the adjacency matrix are inconsistent with the number of agents"

    time = range(1, steps+1) 
    x_ini = np.copy(x)
    X = np.zeros((steps+1, N))
    X[0] = x_ini

    for i in time:
        dseta = np.random.normal(mu, sigma, N)
        x_aux = x*dseta*a 
        shares = Adj * np.tile(x_aux, (N,1)) #Multiplicación elemento a elemento con x como matriz
        mean = np.divide(shares.sum(axis=1), Adj.sum(axis=1), where=Adj.sum(axis=1) != 0) # Repartimos sólo con los vecinos
        x = x * dseta*(1 - a) + mean
        X[i] = x
            
    return X

"""
function: _evolve_network_defection(Adj, x, a, mu, sigma, steps = 1000)

Evolution of the Stochastic Multiplicative Growth process compared to the defective case in a network.
This evolution function considers that all agents play for the entire evolution period.
It will evolve the system according to the sharing parameters as well as complete
defection using the same random numbers. This way we can compare the evolution in 
full defection mode compared to a cooperative case.
The sharing parameter of an agent is applied to every one of its links, so that the total amount
of value it shares is its share parameter times its number of links.

Inputs:
Adj: Adjecency Matrix.
x: Initial values of the agents
a: Sharing parameters of each agent
mu: mean of the normal distribution used for the stochasticity
sigma: Standard deviation of the normal distribution used for the stochasticity
steps: Number of time steps to consider.

Returns:
X_coop: [steps, len(x)] array. Value of the agents along the cooperative evolution
X_def: [steps, len(x)] array. Value of the agents along the defective evolution
"""
def _evolve_network_defection(Adj, x, a, mu, sigma, steps = 1000):
    assert len(x) == len(a), "Número de agentes inconsistente \n Inconsistent number of agents"
    N = len(x)
    assert (N,N) == Adj.shape, "Dimensiones de la matriz de adyacencia inconsistentes con el número de agentes \n The dimensions of the adjacency matrix are inconsistent with the number of agents"

    time = range(1, steps+1) 
    x_def = np.copy(x)
    x_ini = np.copy(x)

    X_coop = np.zeros((steps+1, N))
    X_def = np.zeros((steps+1, N))
    X_coop[0], X_def[0] = x_ini, x_ini

    for i in time:
        dseta = np.random.normal(mu, sigma, N)
        x_aux = x*dseta*a 
        shares = Adj * np.tile(x_aux, (N,1)) #Multiplicación elemento a elemento con x como matriz
        mean = np.divide(shares.sum(axis=1), Adj.sum(axis=1), where=Adj.sum(axis=1) != 0) # Repartimos sólo con los vecinos
        x = x * dseta*(1 - a) + mean
        x_def = x_def*dseta

        X_coop[i] = x
        X_def[i] = x_def
            
    return X_coop, X_def



"""
function: evolve(case, x, a, mu, sigma, steps = 1000)

Calls the evolution functions with the corresponding
parameters depending on the specified case.
"""
def evolve(case, x, a, mu, sigma, steps = 1000):
    if case == 'cooperation':
        return _evolve_cooperation(x, a, mu, sigma, steps)
    if case == 'defection':
        return _evolve_defection(x, a, mu, sigma, steps)

    raise ValueError(f'{case} is not a valid case')


""" 
function: get_growth(X)

Calculates the logarithmic growth rate of the agent along its evolution

Inputs:
X: [steps, len(x)] array. Value of the agents along the evolution

Returns:
Gamma: Logarithmic growth rate of the agent along the evolution

"""
def get_growth(X):
    steps, N = X.shape
    time = range(1, steps) 
    x_ini = np.copy(X[0,:])
    Gamma = np.zeros((steps, N))

    for i in time:
        Gamma[i] = np.log(X[i,:]/x_ini)/i
    return Gamma


if __name__=="__main__":
    adj_matrix = np.array([
    [0,1,1,1,1,1,1],
    [1,0,1,0,0,0,0],
    [1,1,0,1,0,0,0],
    [1,0,1,0,0,0,0],
    [1,0,0,0,0,1,0],
    [1,0,0,0,1,0,0],
    [1,0,0,0,0,0,0],
    ])
    N = 7
    x0 = 1
    a_i = 0.2
    x_ini = np.ones(N)*x0
    a = np.ones(N) * a_i
    steps = int(1e3)
    mu = 1
    sigma = 0.1 
    X = _evolve_network_defection(adj_matrix, x_ini, a, mu, sigma)
