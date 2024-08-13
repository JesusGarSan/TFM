"""
This module contains all the functions needed to evolve the agent models.
It takes charge of all the underlaying mathematics
"""

import numpy as np
"""
function: _test_agent_number(N, x, a, mu, sigma)

This function checks that the dimensions of the provided arrays is correct.
If the value provided for one of the arrays is a float value, the funciton will
convert it into an array of the adequate dimension. 
This funcion will be called before starting an evolution process.

Inputs: 
N: Number of agents
x: Initial value of the agents
a: Sharing parameter of the agents
mu: Mean of the random distribution of the agent 
sigma: Standard deviation of the random distribution of the agent 

"""
def _test_agent_number(N, x, a, mu, sigma):
    assert type(N) == int, "The number of agents (N) must be an integer"
    if type(x) == float: x = np.ones(N)*x
    if type(a) == float: a = np.ones(N)*a
    if type(mu) == float: mu = np.ones(N)*mu
    if type(sigma) == float: sigma = np.ones(N)*sigma

    assert len(x) == N, "x must have the same dimension as the number of agents"
    assert len(a) == N, "a must have the same dimension as the number of agents"
    assert len(mu) == N, "mu must have the same dimension as the number of agents"
    assert len(sigma) == N, "sigma must have the same dimension as the number of agents"

    return N, x, a, mu, sigma


"""
function: _evolve_cooperation(x, a, mu=1, sigma=0.1, steps = 1000)

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
np.zeros((steps+1, N)): We do this for consistency with the defection cases
"""
def _evolve_cooperation(N, x, a, mu, sigma, steps):
    time = range(1, steps+1) 
    x_ini = np.copy(x)

    X = np.zeros((steps+1, N))
    X[0] = x_ini

    for i in time:
        dseta = np.random.normal(mu, sigma)
        x = x * dseta*(1 - a) + np.mean(a*x*dseta)
        X[i] = x
            
    return X, np.zeros((steps+1, N))
def _evolve_cooperation(N, x, a, mu, sigma, steps, gen_freq):
    time = range(1, steps+1) 
    x_ini = np.copy(x)

    X = np.zeros((steps+1, N))
    X[0] = x_ini

    for i in time:
        if i%gen_freq == 0:
            x, a = next_gen(x, a, 0.5)
            print(np.sort(a)) 
        dseta = np.random.normal(mu, sigma)
        x = x * dseta*(1 - a) + np.mean(a*x*dseta)
        X[i] = x
            
    return X, np.zeros((steps+1, N))

"""
function: _evolve_defection(x, a, mu=1, sigma=0.1, steps = 1000)

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
def _evolve_defection(N, x, a, mu, sigma, steps):
    time = range(1, steps+1) 
    x_def = np.copy(x)
    x_ini = np.copy(x)

    X_coop = np.zeros((steps+1, N))
    X_def = np.zeros((steps+1, N))
    X_coop[0], X_def[0] = x_ini, x_ini
    for i in time:
        dseta = np.random.normal(mu, sigma)
        x = x * dseta*(1 - a) + np.mean(a*x*dseta)
        x_def = x_def*dseta

        X_coop[i] = x
        X_def[i] = x_def
            
    return X_coop, X_def

"""
function: evolve_network(Adj, x, a, mu=1, sigma=0.1, steps = 1000)

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
np.zeros((steps+1, N)): We do this for consistency with the defection cases
"""
def _evolve_network_cooperation(Adj, N, x, a, mu, sigma, steps):
    assert (N,N) == Adj.shape, "Dimensiones de la matriz de adyacencia inconsistentes con el número de agentes \n The dimensions of the adjacency matrix are inconsistent with the number of agents"

    time = range(1, steps+1) 
    x_ini = np.copy(x)
    X = np.zeros((steps+1, N))
    X[0] = x_ini

    for i in time:
        dseta = np.random.normal(mu, sigma)
        x_aux = x*dseta*a 
        shares = Adj * np.tile(x_aux, (N,1)) #Multiplicación elemento a elemento con x como matriz
        mean = np.divide(shares.sum(axis=1), Adj.sum(axis=1), where=Adj.sum(axis=1) != 0) # Repartimos sólo con los vecinos
        x = x * dseta*(1 - a) + mean
        X[i] = x
            
    return X, np.zeros((steps+1, N))

"""
function: _evolve_network_defection(Adj, x, a, mu=1, sigma=0.1, steps = 1000)

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
def _evolve_network_defection(Adj, N, x, a, mu, sigma, steps):
    assert (N,N) == Adj.shape, "Dimensiones de la matriz de adyacencia inconsistentes con el número de agentes \n The dimensions of the adjacency matrix are inconsistent with the number of agents"

    time = range(1, steps+1) 
    x_def = np.copy(x)
    x_ini = np.copy(x)

    X_coop = np.zeros((steps+1, N))
    X_def = np.zeros((steps+1, N))
    X_coop[0], X_def[0] = x_ini, x_ini

    for i in time:
        dseta = np.random.normal(mu, sigma)
        x_aux = x*dseta*a 
        shares = Adj * np.tile(x_aux, (N,1)) #Multiplicación elemento a elemento con x como matriz
        mean = np.divide(shares.sum(axis=1), Adj.sum(axis=1), where=Adj.sum(axis=1) != 0) # Repartimos sólo con los vecinos
        x = x * dseta*(1 - a) + mean
        x_def = x_def*dseta

        X_coop[i] = x
        X_def[i] = x_def
            
    return X_coop, X_def



"""
function: evolve(case, x, a, mu=1, sigma=0.1, steps = 1000)

Calls the evolution functions with the corresponding
parameters depending on the specified case.
"""
def evolve(N=2, x=100.0, a=0.5, mu=1.0, sigma=0.1, defection = True, steps=int(1e4), **kwargs):
    N, x, a, mu, sigma = _test_agent_number(N, x, a, mu, sigma)
    network = "Adj" in kwargs
    if network: Adj = kwargs['Adj']

    if defection and network:
        return _evolve_network_defection(Adj, N, x, a, mu, sigma, steps)
    if defection and not network:
        return _evolve_defection(N, x, a, mu, sigma, steps)
    if not defection and network:
        return _evolve_network_cooperation(Adj, N, x, a, mu, sigma, steps)
    if not defection and not network:
        return _evolve_cooperation(N, x, a, mu, sigma, steps)

    raise ValueError(f'Invalid parameters entered')


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


"""
function: repopulate()
"""
def next_gen(x, a, filter=0.5):
    # kill
    death_toll = int(len(x) * filter)
    id_sorted = np.argsort(x)  # Ordena los índices según los valores en x

    x = x[id_sorted]  # Ordena x según los índices
    a = a[id_sorted]  # Ordena a según los índices

    # New gen
    survivors = x[death_toll:]  # Valores de x que sobrevivirán
    survivors_a = a[death_toll:]  # Valores correspondientes en a

    parents = (np.random.rand(death_toll) * len(survivors)).astype(int)
    childs_a = survivors_a[parents] * np.random.uniform(1/1.1, 1.1)
    x[:death_toll] = survivors[parents]
    a[:death_toll] = childs_a

    return x, a
        
"""
function: selection()

We use this function to discard the worst performing agents every few steps

"""

def selection():
    return



"""
WIP
function optimize_sharing()

Calculates the optimal sharing parameter of an agent to maximize its relative growth rate .
"""

def optimize_sharing(x_ini, a, delta_a = 0.1, precision = 0.01, max_attempts = 5,
                     max_steps = int(1e4), agent_id=0, cpus = 2, verbose = False):
    from simulate import simulate, gamma_stats
    old_gamma_rel = 0
    delta_gamma = 0
    trend = +1
    attempts = 1

    A = np.zeros(max_steps)
    Delta_gamma = np.zeros(max_steps)
    Trend = np.zeros(max_steps)
    Delta_a = np.zeros(max_steps)
    for m in range(max_steps):
        _, _, Gammas_coop, Gammas_def=  simulate(x_ini, a, 1.0, 0.1, M = 154, cpus=cpus)    
        _, _,gamma_rel, _ = gamma_stats(Gammas_coop, Gammas_def)
        delta_gamma = gamma_rel-old_gamma_rel
        if verbose: print(f"Step {m} with a_{agent_id} = {a[agent_id]}, delta_a = {delta_a}, delta_gamma = {delta_gamma}")
        

        # If the change in a is significative, we keep going
        if delta_a > precision:
            old_gamma_rel = gamma_rel
            # If the trend is favorable, we keep it
            if delta_gamma > 0:
                # delta_a *= 1.25
                pass
            # If the trend is detrimental, we reverse it and reduce the step
            if delta_gamma < 0:
                trend *=-1
                delta_a *= 0.75

            a[agent_id] += trend*delta_a
            a[agent_id] = np.clip(a[agent_id], 0, 1)
            attempts = 1
        
        # If the change is no significative, we try again
        if delta_a < precision:
            
            # delta_a *= 0.5
            # delta_a = precision
            # If the change stays small long enough, we take that as our result
            attempts +=1
            if attempts > max_attempts:
                return a[agent_id], m, A, Delta_gamma, Trend, Delta_a
            if delta_gamma < 0:
                delta_a *= 1.25
            
        A[m] = a[agent_id]
        Delta_gamma[m] = delta_gamma
        Trend[m] = trend
        Delta_a[m] = delta_a
    # If the result is not found within the maximum number of steps allowed, we return what we got
    return a[agent_id], m, A, Delta_gamma, Trend, Delta_a



if __name__=="__main__":
    N = 6
    x = np.ones(N) * 10000.0
    a = np.linspace(0.8, 0.9, N)
    print(a) 

    gen_freq = 100 # Every 1000 steps a new generation happens

    X, _ = _evolve_cooperation(N, x, a, 1.005, 0.1, int(2e4), gen_freq )
    print(X[-1])
    import plot
    plot.evolution(X)

    quit()

    evolve()
    quit()
    import plot
    import matplotlib.pyplot as plt
    from simulate import *
    np.random.seed(124)
    N = 10
    x = np.ones(N)
    a = np.ones(N)*0.5
    a[0] = 0.3
    optimal_a, steps, A, Delta_gamma, Trend, Delta_a = optimize_sharing(x, a, max_steps=30, cpus=7, verbose=True)

    print(optimal_a, steps)

    plt.plot(A, label=r"Optimal Sharing parameter")
    plt.plot(Delta_gamma, label=r"$\Delta \gamma$")
    plt.plot(Trend, label ='trend')
    plt.plot(Delta_a, label=r'$\Delta a$')
    plt.xlim(0, steps-1)
    plt.legend()
    plt.show()
    quit()


    trend = +1
    delta = 0.01

    M = 100
    A = np.zeros(M)
    Trend = np.zeros(M)
    Gammas_rel = np.zeros(M)

    N = 2
    x = np.ones(N)
    a = np.ones(N)*0.5
    a[0] = .45
    A = np.zeros(100)
    A[0] = a[0]
    # np.random.seed(123)
    old_gamma_rel = 0
    delta_gamma = 0
    for i in range(M):
        print(f"Iteration {i}, {round(a[0], 2)}, {old_gamma_rel}, {delta_gamma}")
        # X, X_def = evolve(x, a, 1.0, 0.1, steps=10000)
        _, _, Gammas_coop, Gammas_def=  simulate(x, a, 1, 0.1, M = 50, cpus=7)

        gamma_rel, _ = gamma_stats(Gammas_coop, Gammas_def)
        delta_gamma = gamma_rel-old_gamma_rel

        if delta_gamma > 0: 
            a[0] += trend*delta
        if delta_gamma < 0: 
            trend *= -1
            a[0] += trend*delta

        a[0] = np.clip(a[0], 0, 1)

        old_gamma_rel = gamma_rel
        A[i] = a[0]
        Trend[i] = trend
        Gammas_rel[i] = gamma_rel
    
    plt.plot(A)
    plt.show(block=False)
    
    plt.plot(Trend)
    plt.show(block=False)
    
    plt.plot(Gammas_rel)
    plt.show(block=False)

    input()


    # plt.plot(Gamma_coop - Gamma_def)
    # plt.show(block=False)
    # print(Gamma_def.shape)
    # input()

    quit()
    Adj = np.array([
    [0,1,1,1,1,1,1],
    [1,0,1,0,0,0,0],
    [1,1,0,1,0,0,0],
    [1,0,1,0,0,0,0],
    [1,0,0,0,0,1,0],
    [1,0,0,0,1,0,0],
    [1,0,0,0,0,0,0],
    ])
    N = 2
    x0 = 100
    a_i = 0.2
    x_ini = np.ones(N)*x0
    a = np.ones(N) * a_i
    steps = int(1e4)
    mu = 1
    sigma = 0.1 
    X, X_def = evolve(x_ini, a, defection=False)
    # X=X_def
    print(X.shape)
    plot.evolution(X)