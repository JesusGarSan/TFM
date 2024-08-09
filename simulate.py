"""
This module contains all the functions needed to simulate the
evolution of agent models, saving the results obtained.
"""

from evolution import *

"""
function: simulate(x_ini, a, mu=1, sigma=0.1, defection = True, steps = int(1e4), M=1, plot=False, verbose=False, **kwargs)

Runs simulations of the evolution of the Stochastic Multiplicative Growth process
Depending on its value sit will run simulations comparing with the defective case or not and considering a network or not

Inputs:
x_ini: Initial values of the agents
a: Sharing parameters of each agent
mu: mean of the normal distribution used for the stochasticity
sigma: Standard deviation of the normal distribution used for the stochasticity
defection: Boolean. Determines wether the comparison with defection will be performed or not
steps: Number of time steps to consider.
M: Number of simulations to rur
plot: Boolean. Determines wether a plot will be created or not
verbose: Boolean. Determines wether the step fo the process will be shown in the terminal or not
**kwargs: Adj if network

Returns:
X_coop: [len(x_ini), M] array. Last value of the agents after every cooperative simulation
X_def: [len(x_ini), M] array. Last value of the agents after every defective simulation
Gammas_coop: [len(x_ini), M] array. Last value of the logarithmic growth rate in the cooperative case
Gammas_def: [len(x_ini), M] array. Last value of the logarithmic growth rate in the defective case
"""


def simulate(x_ini, a, mu=1, sigma=0.1, defection = True, steps = int(1e4), M=1, plot=False, verbose=False, **kwargs):
    assert len(x_ini) == len(a), "Número de agentes inconsistente \n Inconsistent number of agents"
    N = len(x_ini)
    X_coop, X_def, Gammas_coop, Gammas_def = np.array([np.zeros((N,M))] * 4)

    for m in range(M):
        if verbose: print(f"Running simulation {m+1}...")
        if plot:
            import plot
            x_coop, x_def = evolve(x_ini, a, mu, sigma, defection, steps, **kwargs)
            plot.evolution(x_coop)
            plot.evolution(x_def)
        if not plot:
            x_coop, x_def = evolve(x_ini, a, mu, sigma, defection, steps, **kwargs)

        X_coop[:,m] = x_coop[-1, :]
        Gammas_coop[:,m] = get_growth(x_coop)[-1]

        X_def[:,m] = x_def[-1, :]
        Gammas_def[:,m] = get_growth(x_def)[-1]

    return X_coop, X_def, Gammas_coop, Gammas_def


"""
This function generates the data required to replicate Figure 1 in the suplemmental material of Lorenzo's Paper.
To be compatible with calls via parallelization, its parameters are taken as a tuple
"""
def fig_1_simulation(params = (2, 0.1, 10, int(1e4), 1.0, 0.5, 1.0, 0.1, False, False)):
    N, a_1, M, steps, x_ini, a_i, mu, sigma, save, verbose = params
    if verbose: print(f"Running for {N} agents & share parameter {round(a_1, 2)}...")
    x = np.ones(N) * x_ini
    a = np.ones(N) * a_i
    a[0] = a_1

    _, _, Gammas, Gammas_def = simulate(x,a,mu,sigma, True, steps, M)

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

def fig_1_simulation_parallel(N_array, a_1_array, M = 10, steps=int(1e4), x_ini=1.0, a_i=0.5, mu = 1.0, sigma=0.1, save = False, verbose = False, cores = 7):
    import multiprocessing
    parameters = []
    for N in N_array:
        for a_1 in a_1_array:
            parameters.append((N, a_1, M, steps, x_ini, a_i, mu, sigma, save, verbose))

    with multiprocessing.Pool(cores) as pool:
        pool.map(fig_1_simulation, parameters)



if __name__=="__main__":

    N_array = [2,3,4,6,10]
    a_1_array = np.arange(0, 1.5, 0.02)[1:]
    fig_1_simulation_parallel(N_array, a_1_array, M=500, steps =int(1e4), save = True, verbose=True, cores = 8)
    quit()

    N = [2,3,4,6,10]
    a = np.arange(0, 1.5, 0.02)[1:]
    fig_1_data(N, a, M = 100, steps = 10000, save=True, verbose=True)
    
    quit()
    A = np.array([
        [1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,1,1,1,0,0],

        [0,0,0,0,0,0,0,0,1,1],
        [0,0,0,0,0,0,0,0,1,1],
    ])
        
    N = len(A)
    x = np.ones(N)
    a = np.ones(N)*0.1

    _, _, Gamma_coop, Gamma_def = simulate(x, a, M= 3, steps = int(1e4), verbose=True, Adj = A)

    import plot
    plot.growth_rates(Gamma_coop, Gammas_def=Gamma_def)


