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
function: evolve(case, x, a, mu, sigma, steps = 1000)

Calls the function _evolve_cooperation or _evolve_defection with the corresponding
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
gamma: Logarithmic growth rate of the agent

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
    pass

    N = 3
    x0 = 1
    a_i = 0.2
    x_ini = np.ones(N)*x0
    a = np.ones(N) * a_i
    steps = int(1e3)
    mu = 1
    sigma = 0.1  

    X_coop, X_def = evolve('defection', x_ini, a, mu, sigma, steps=1000)
    Gamma = get_growth(X_coop)
    print(Gamma[-1])
    Gamma = get_growth(X_def)
    print(Gamma[-1])