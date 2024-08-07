"""
This module contains all the functions needed to simulate the
evolution of agent models, saving the results obtained.
"""

from evolution import *

"""
function: _simulate_cooperation(x_ini, a, mu=1, sigma=0.1, steps = int(1e4), M=1, plot=False, verbose=False)

Runs simulations of the evolution of the Stochastic Multiplicative Growth process

Inputs:
x_ini: Initial values of the agents
a: Sharing parameters of each agent
mu: mean of the normal distribution used for the stochasticity
sigma: Standard deviation of the normal distribution used for the stochasticity
steps: Number of time steps to consider.
M: Number of simulations to rur
plot: Boolean. Determines wether a plot will be created or not
verbose: Boolean. Determines wether the step fo the process will be shown in the terminal or not

Returns:
X_coop: [len(x_ini), M] array. Last value of the agents after every simulation
Gammas: [len(x_ini), M] array. Last value of the logarithmic growth rate
"""
def _simulate_cooperation(x_ini, a, mu=1, sigma=0.1, steps = int(1e4), M=1, plot=False, verbose=False):
    assert len(x_ini) == len(a), "Número de agentes inconsistente \n Inconsistent number of agents"
    N = len(x_ini)
    X_coop, Gammas = np.array([np.zeros((N,M))] * 2)

    # Simulations
    for m in range(M):
        if verbose: print(f"Running simulation {m}...")
        if plot:
            import plot
            x_coop = evolve('cooperation', x_ini, a, mu, sigma, steps = steps)
            plot.evolution(x_coop)
        if not plot:
            x_coop = evolve('cooperation', x_ini, a, mu, sigma, steps = steps)
        
        X_coop[:,m] = x_coop[-1, :]
        Gammas[:,m] = get_growth(x_coop)[-1]

    return X_coop, Gammas

"""
function: _simulate_defection(x_ini, a, mu=1, sigma=0.1, steps = int(1e4), M=1, plot=False, verbose=False)

Runs simulations of the evolution of the Stochastic Multiplicative Growth process
compared to the defective case.

Inputs:
x_ini: Initial values of the agents
a: Sharing parameters of each agent
mu: mean of the normal distribution used for the stochasticity
sigma: Standard deviation of the normal distribution used for the stochasticity
steps: Number of time steps to consider.
M: Number of simulations to rur
plot: Boolean. Determines wether a plot will be created or not
verbose: Boolean. Determines wether the step fo the process will be shown in the terminal or not

Returns:
X_coop: [len(x_ini), M] array. Last value of the agents after every cooperative simulation
X_def: [len(x_ini), M] array. Last value of the agents after every defective simulation
Gammas_coop: [len(x_ini), M] array. Last value of the logarithmic growth rate in the cooperative case
Gammas_def: [len(x_ini), M] array. Last value of the logarithmic growth rate in the defective case
"""
def _simulate_defection(x_ini, a, mu=1, sigma=0.1, steps = int(1e4), M=1, plot=False, verbose=False):
    assert len(x_ini) == len(a), "Número de agentes inconsistente \n Inconsistent number of agents"
    N = len(x_ini)
    X_coop, X_def, Gammas_coop, Gammas_def = np.array([np.zeros((N,M))] * 4)

    # Simulations
    for m in range(M):
        if verbose: print(f"Running simulation {m}...")
        if plot:
            import plot
            x_coop, x_def = evolve('defection', x_ini, a, mu, sigma, steps = steps)
            plot.evolution(x_coop)
            plot.evolution(x_def)
        if not plot:
            x_coop, x_def = evolve('defection', x_ini, a, mu, sigma, steps = steps)

        X_coop[:,m] = x_coop[-1, :]
        Gammas_coop[:,m] = get_growth(x_coop)[-1]

        X_def[:,m] = x_def[-1, :]
        Gammas_def[:,m] = get_growth(x_def)[-1]

    return X_coop, X_def, Gammas_coop, Gammas_def

"""
function: simulate(case, x_ini, a, mu=1, sigma=0.1, steps = int(1e4), M=1, plot=False, verbose=False)

Calls the function _simulate_cooperation or _simulate_defection with the corresponding
parameters depending on the specified case.
"""
def simulate(case, x_ini, a, mu=1, sigma=0.1, steps = int(1e4), M=1, plot=False, verbose=False):
    if case == 'cooperation':
        return _simulate_cooperation(x_ini, a, mu, sigma, steps, M, plot, verbose)
    if case == 'defection':
        return _simulate_defection(x_ini, a, mu, sigma, steps, M, plot, verbose)

    raise ValueError(f'{case} is not a valid case')

"""
This function generates the data required to replicate Figure 1 in the suplemmental material of Lorenzo's Paper.
"""
def fig_1_data(N_array, a_1_array, M = 10, steps=int(1e4), x_ini=1.0, a_i=0.5, mu = 1.0, sigma=0.1, save = False, verbose = False):
    for N in N_array:
        x = np.ones(N) * x_ini
        for a_1 in a_1_array:
            if verbose: print(f"Running for {N} agents & share parameter {round(a_1, 2)}...")
            a = np.ones(N) * a_i
            a[0] = a_1

            _, _, Gammas, Gammas_def = simulate('defection',x,a,mu,sigma,steps,M)

            Gammas_rel = (Gammas[0,:] - Gammas_def[0,:]) * 100
            gamma = np.mean(Gammas[0,:]) * 100
            gamma_def = np.mean(Gammas_def[0,:]) * 100
            gamma_rel = np.mean(Gammas_rel)
            error = np.std(Gammas_rel)/np.sqrt(len(Gammas_rel))
            rel_error = error/gamma_rel

            if save:
                import csv
                row = [N, x_ini, a_i, a_1, mu, sigma, steps, M, gamma, gamma_def, gamma_rel, error, rel_error]
                with open('./data/relative_long_term_growth_rate.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)
    return

if __name__=="__main__":
    N = [2,3,4,6,10]
    a = np.arange(0, 1.5, 0.02)[1:]
    fig_1_data(N, a, M = 100, steps = 10000, save=True, verbose=True)
    
    # x_ini = np.ones(3)
    # a = np.ones(3)*0.2
    # res = simulate('defection', x_ini, a , M = 2   )

    # print(res[-1])
    # print(res[-2])

