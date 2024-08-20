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
    if type(x)     == float: x = np.ones(N)*x
    if type(a)     == float: a = np.ones(N)*a
    if type(mu)    == float: mu = np.ones(N)*mu
    if type(sigma) == float: sigma = np.ones(N)*sigma

    assert len(x)     == N, "x must have the same dimension as the number of agents"
    assert len(a)     == N, "a must have the same dimension as the number of agents"
    assert len(mu)    == N, "mu must have the same dimension as the number of agents"
    assert len(sigma) == N, "sigma must have the same dimension as the number of agents"

    return N, x, a, mu, sigma

"""
function: update_a

Updates the value of the sharing parameter a depending on the value of the agents x.

inputs:
Inputs:
a: Sharing parameters of each agent
x: Values of the agents
mode: Type of regime to consider when updating a
exponent: Exponent to use in the regime. It measures the magnitud of the regime

Returns:
a: new sharing aprameters of the agents
"""

def update_a(a, x, mode, exponent):
    X_total = np.sum(x)
    X_rel = x/X_total
    if mode =='greedy':   return (1-X_rel)**exponent
    if mode =='generous': return (X_rel)**exponent

    from warnings import warn
    warn(f"Warning: {mode} does not correspond to any of the modes implemented in update_a. Returning original a array")
    return a


"""
function: evolve(x, a, mu=1, sigma=0.1, steps = 1000)

Evolution of the Stochastic Multiplicative Growth process compared to the defective case.
This evolution function considers that all agents play for the entire evolution period.
It will evolve the system according to the sharing parameters as well as complete
defection using the same random numbers. This way we can compare the evolution in 
full defection mode compared to a cooperative case.

Inputs:
N: Number of agents
x: Initial values of the agents
a: Sharing parameters of each agent
mu: mean of the normal distribution used for the stochasticity
sigma: Standard deviation of the normal distribution used for the stochasticity
steps: Number of time steps to consider.
new_a: Parameters to consider if a is going to be updated each step
new_generation: Parameters to consider if biological evolution is going to be considered

Returns:
X_coop: [steps, len(x)] array. Value of the agents along the cooperative evolution
X_def: [steps, len(x)] array. Value of the agents along the defective evolution
"""
def evolve(N=2, x=100.0, a=0.5, mu=1.0, sigma=0.1, steps=int(1e4), new_a = False, new_generation = False):
    N, x, a, mu, sigma = _test_agent_number(N, x, a, mu, sigma)
    if new_generation: gen_steps = new_generation[0]
    time = range(1, steps+1) 
    x_def = np.copy(x)
    x_ini = np.copy(x)

    stats = {
        "X_coop":      np.zeros((steps+1, N)),
        "X_def":       np.zeros((steps+1, N)),
        "a_array":     np.zeros((steps+1, N)),
        "mu_array":    np.zeros((steps+1, N)),
        "sigma_array": np.zeros((steps+1, N)),
    }
    stats["X_coop"][0], stats["X_def"][0] = x_ini, x_def
    stats["a_array"][0], stats["mu_array"][0], stats["sigma_array"][0] = a, mu, sigma

    for i in time:
        if new_a: a = update_a(*((a,x) + new_a))
        if new_generation and i%gen_steps == 0:
            x, a, mu, sigma = next_gen(*(x,a,mu,sigma)+new_generation[1:])
        dseta = np.random.normal(mu, sigma)
        x = x * dseta*(1 - a) + np.mean(a*x*dseta)
        x_def = x_def*dseta

        stats["X_coop"][i] = x
        stats["X_def"][i] = x_def
        stats["a_array"][i], stats["mu_array"][i], stats["sigma_array"][i] = a, mu, sigma

    return stats




"""
function: evolve_network(Adj, x, a, mu=1, sigma=0.1, steps = 1000)

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
new_a: Parameters to consider if a is going to be updated each step
new_generation: Parameters to consider if biological evolution is going to be consideredr.

Returns:
X_coop: [steps, len(x)] array. Value of the agents along the cooperative evolution
X_def: [steps, len(x)] array. Value of the agents along the defective evolution
"""
def evolve_network(Adj, N=2, x=100.0, a=0.5, mu=1.0, sigma=0.1, steps=int(1e4), new_a = False, new_generation = False):
    N, x, a, mu, sigma = _test_agent_number(N, x, a, mu, sigma)
    if new_generation: gen_steps = new_generation[0]
    assert (N,N) == Adj.shape, "Dimensiones de la matriz de adyacencia inconsistentes con el número de agentes \n The dimensions of the adjacency matrix are inconsistent with the number of agents"

    time = range(1, steps+1) 
    x_def = np.copy(x)
    x_ini = np.copy(x)

    stats = {
        "X_coop":      np.zeros((steps+1, N)),
        "X_def":       np.zeros((steps+1, N)),
        "a_array":     np.zeros((steps+1, N)),
        "mu_array":    np.zeros((steps+1, N)),
        "sigma_array": np.zeros((steps+1, N)),
    }
    stats["X_coop"][0], stats["X_def"][0] = x_ini, x_def
    stats["a_array"][0], stats["mu_array"][0], stats["sigma_array"][0] = a, mu, sigma

    for i in time:
        if new_a: a = update_a(*((a,x) + new_a))
        if new_generation and i%gen_steps == 0:
            x, a, mu, sigma = next_gen(*(x,a,mu,sigma)+new_generation[1:])
        dseta = np.random.normal(mu, sigma)
        x_aux = x*dseta*a 
        shares = Adj * np.tile(x_aux, (N,1)) #Multiplicación elemento a elemento con x como matriz
        mean = np.divide(shares.sum(axis=1), Adj.sum(axis=1), where=Adj.sum(axis=1) != 0) # Repartimos sólo con los vecinos
        x = x * dseta*(1 - a) + mean
        x_def = x_def*dseta

        stats["X_coop"][i] = x
        stats["X_def"][i] = x_def
        stats["a_array"][i], stats["mu_array"][i], stats["sigma_array"][i] = a, mu, sigma

    return stats



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
function: next_gen()
"""
def next_gen(x, a, mu, sigma, heritage=['x', 'a', 'mu', 'sigma', 'gamma'], variance = [0.1, 0.1], filter=0.5, mutation = 0.0):

    # Best performing agents
    id_sorted = np.argsort(x)  
    x     = x[id_sorted]  
    a     = a[id_sorted]  
    mu    = mu[id_sorted]  
    sigma = sigma[id_sorted]  

    # Survivors
    death_toll = int(len(x) * filter)
    survivors       = x[death_toll:] 
    survivors_a     = a[death_toll:]  
    survivors_mu    = mu[death_toll:]  
    survivors_sigma = sigma[death_toll:]  


    # Survivors that produc offspring
    parents = (np.random.rand(death_toll) * len(survivors)).astype(int)

    # Hereditary characteristics of the offspring
    if 'x' in heritage: x[:death_toll] = survivors[parents]

    if 'a' in heritage:a[:death_toll] = survivors_a[parents] * np.random.uniform(1-variance[0], 1+variance[1]) + np.random.normal(size=len(parents))*mutation
    else: a[:death_toll] = survivors_a[parents]
    a = np.clip(a, 0, 1)

    if 'mu' in heritage: mu[:death_toll] = survivors_mu[parents] * np.random.uniform(1-variance[0], 1+variance[1])
    else: mu[:death_toll] = survivors_mu[parents]

    if 'sigma' in heritage: sigma[:death_toll] = survivors_sigma[parents] * np.random.uniform(1-variance[0], 1+variance[1])
    else: sigma[:death_toll] = survivors_sigma[parents]


    return x, a, mu, sigma


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

    N = 20
    x = np.ones(N) * 10000.0
    a = np.linspace(0.1, 0.1, N)
    # a[-1] = 0.5

    import matplotlib.pyplot as plt
    np.random.seed(123)
    stats = evolve(N, x, a, mu=1.00, steps= int(1e5), new_generation=(int(1e2), ['a'], [0.01, 0.1], 0.2, ))
    print(f" Total x: {round(np.sum(stats["X_coop"][-1, :]), 8)}. Average a: {np.mean(stats["a_array"][-1])}")
    fig, ax = plt.subplots()
    plt.plot(stats["a_array"][101:, :], lw=0.05)
    plt.plot(np.mean(stats["a_array"][101:, :], axis=1), lw=1)
    plt.show(block=False)
    fig, ax = plt.subplots()
    plt.plot(np.mean(stats["X_coop"], axis=1))
    plt.show(block=False)

    np.random.seed(123)
    stats = evolve(N, x, a, mu=1.00, steps= int(1e5), new_generation=(int(1e2), ['a'], [0.1, 0.1], 0.2, ))
    print(f" Total x: {round(np.sum(stats["X_coop"][-1, :]), 8)}. Average a: {np.mean(stats["a_array"][-1])}")
    fig, ax = plt.subplots()
    plt.plot(stats["a_array"][101:, :], lw=0.05)
    plt.plot(np.mean(stats["a_array"][101:, :], axis=1), lw=1)
    plt.show(block=False)
    fig, ax = plt.subplots()
    plt.plot(np.mean(stats["X_coop"], axis=1))
    plt.show(block=False)
    input()
