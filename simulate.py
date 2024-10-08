"""
This module contains all the functions needed to simulate the
evolution of agent models, saving the results obtained.
It includes parallelization features
"""
import numpy as  np
from evolution import evolve, get_growth

"""
function: _simulate_single(x_ini, a, mu=1, sigma=0.1, defection = True, steps = int(1e4), M=1, plot=False, verbose=False, **kwargs)

Runs simulations of the evolution of the Stochastic Multiplicative Growth process
Depending on its value sit will run simulations comparing with the defective case or not and considering a network or not
It runs on a single CPU

Inputs:
x_ini: Initial values of the agents
a: Sharing parameters of each agent
mu: mean of the normal distribution used for the stochasticity
sigma: Standard deviation of the normal distribution used for the stochasticity
defection: Boolean. Determines wether the comparison with defection will be performed or not
steps: Number of time steps to consider.
M: Number of simulations to run
plot: Boolean. Determines wether a plot will be created or not
verbose: Boolean. Determines wether the step fo the process will be shown in the terminal or not
**kwargs: Adj if network

Returns:
X_coop: [len(x_ini), M] array. Last value of the agents after every cooperative simulation
X_def: [len(x_ini), M] array. Last value of the agents after every defective simulation
Gammas_coop: [len(x_ini), M] array. Last value of the logarithmic growth rate in the cooperative case
Gammas_def: [len(x_ini), M] array. Last value of the logarithmic growth rate in the defective case
"""
def _simulate_single(x_ini, a, mu=1, sigma=0.1, steps = int(1e4), M=10, verbose=False, **kwargs):
    X_coop, X_def, Gammas_coop, Gammas_def = np.array([np.zeros((N,M))] * 4)
    for m in range(M):
        if verbose: print(f"Running simulation {m+1}...")

        X_coop_final, X_def_final, Gammas_coop_final, Gammas_def_final = _simulate(x_ini, a, mu, sigma, steps, **kwargs)

        X_coop[:,m] = X_coop_final
        X_def[:,m] = X_def_final
        Gammas_coop[:,m] = Gammas_coop_final
        Gammas_def[:,m] = Gammas_def_final

    return X_coop, X_def, Gammas_coop, Gammas_def

"""
function: _simulate(x_ini, a, mu, sigma, defection, steps, **kwargs)

Runs one simulation of the evolution of the Stochastic Multiplicative Growth process
Depending on its value sit will run simulations comparing with the defective case or not and considering a network or not
This function is called by evolve to run in parallel to other simulations.

Inputs:
x_ini: Initial values of the agents
a: Sharing parameters of each agent
mu: mean of the normal distribution used for the stochasticity
sigma: Standard deviation of the normal distribution used for the stochasticity
steps: Number of time steps to consider.
**kwargs

Returns:
stats: Diccionary with the data regarding the evolution
"""
def _simulate(N, x_ini, a, mu, sigma, steps, **kwargs):
    return evolve(N, x_ini, a, mu, sigma, steps, **kwargs)

"""
function: simulate(x_ini, a, mu=1, sigma=0.1, defection=True, steps=int(1e4), M=10, cpus = 2, **kwargs)

This functions runs several evolution simulations in parallel. Depending on the number of CPUs employed
the function _simulate or _simulate_single will be called.

Inputs:
x_ini: Initial values of the agents
a: Sharing parameters of each agent
mu: mean of the normal distribution used for the stochasticity
sigma: Standard deviation of the normal distribution used for the stochasticity
defection: Boolean. Determines wether the comparison with defection will be performed or not
steps: Number of time steps to consider.
M: Number of simulations to run
cpus: Number of CPUs to use
**kwargs: Adj if network

Returns:
X_coop: [len(x_ini), M] array. Last value of the agents after every cooperative simulation
X_def: [len(x_ini), M] array. Last value of the agents after every defective simulation
Gammas_coop: [len(x_ini), M] array. Last value of the logarithmic growth rate in the cooperative case
Gammas_def: [len(x_ini), M] array. Last value of the logarithmic growth rate in the defective case
"""
def simulate(N, x_ini, a, mu=1, sigma=0.1, steps=int(1e4), M=10, cpus = 2, **kwargs):
    import multiprocessing as mp
    from functools import partial # This is used to pass **kwargs to the function in starmap
    assert cpus <= mp.cpu_count(), "Specified number of CPUs is larger than available."
    if cpus == 1: return _simulate_single(N, x_ini,a,mu,sigma,steps,M,**kwargs)
    # if ('verbose' in kwargs and kwargs['verbose'] == True): verbose = True
    # else: verbose= False 
    verbose = kwargs.get("verbose", False)

    if verbose: print(f"Initializing multiprocessing pool for {M} tasks and {cpus} cpus...")
    pool = mp.Pool(cpus)
    tasks = [(N, x_ini, a, mu, sigma, steps) for _ in range(M)]
    if verbose: print("Running simulations...")
    results = pool.starmap(partial(_simulate, **kwargs), tasks)

    if verbose: print(f"Compiling results...")
    stats_array = [{}]*M
    for m, stats in enumerate(results): stats_array[m] = stats

    # Cerrar el pool de procesos
    pool.close()
    pool.join()

    return stats_array


"""
function: gamma_stats(Gammas_coop, Gammas_def, agent_id =0)

Get the relative gamma and error of a set of a set of gammas from different simulations

Inputs:
Gammas_coop: [N_agents, M_simulations] array. Last values of the growth rate for every agent after every copperative simulation
Gammas_def: [N_agents, M_simulations] array. Last values of the growth rate for every agent after every defective simulation

Returns:
gamma_coop: [N_agents] array. Average across simulations of the last value for the cooperative growth rate of every agent
gamma_def: [N_agents] array. Average across simulations of the last value for the defective growth rate of every agent
gamma_rel: [N_agents] array. Relative growth rate of every agent
error: [N_agents] array. Error of gamma_rel based on the standard deviation of the values.
**kwargs: agent_id if we want to return the values of only one agent
"""
def gamma_stats(Gammas_coop, Gammas_def, **kwargs):
    assert Gammas_coop.shape == Gammas_def.shape, "The dimensions of both growth rates must be the same."
    Gammas_rel = (Gammas_coop - Gammas_def) * 100
    gamma_coop = np.mean(Gammas_coop, axis = 1) * 100
    gamma_def = np.mean(Gammas_def, axis = 1) * 100
    gamma_rel = np.mean(Gammas_rel, axis = 1)
    error = np.std(Gammas_rel, axis = 1)/np.sqrt(Gammas_rel.shape[1])


    if "agent_id" in kwargs:
        id = kwargs["agent_id"]
        return gamma_coop[id], gamma_def[id], gamma_rel[id], error[id]

    return gamma_coop, gamma_def, gamma_rel, error

"""
This function generates the data required to replicate Figure 1 in the suplemmental material of Lorenzo's Paper.
"""
def fig_1_simulation(N_array, a_1_array, M = 10, steps=int(1e4), x_ini=100.0, a_i=0.5, mu = 1.0, sigma=0.1, cpus = 2, save = False, verbose = False):
    for N in N_array:
        for a_1 in a_1_array:
            if verbose: print(f"\nRunning for {N} agents & share parameter {round(a_1, 2)}...")
            a = np.ones(N) * a_i
            a[0] = a_1
            stats_array = simulate(N,x_ini,a,mu,sigma, steps, M, cpus=cpus, verbose = True)
            Gammas_coop = np.array([get_growth(stats["X_coop"])[-1] for stats in stats_array])
            Gammas_def  = np.array([get_growth(stats["X_def"]) [-1] for stats in stats_array])

            gamma_coop, gamma_def, gamma_rel, error = gamma_stats(Gammas_coop, Gammas_def, agent_id = 0)
            rel_error = error/gamma_rel

            if save:
                if verbose: print("Saving results...")
                import csv
                row = [N, x_ini, a_i, a_1, mu, sigma, steps, M, gamma_coop, gamma_def, gamma_rel, error, rel_error]
                with open('./data/fig_1_sup.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)
    return
"""
This function generates the data required to replicate figure 1a of Lorenzo's Paper.
"""
def fig_1a_simulation(N, x_ini, a_i, a_1_array, mu,  sigmas, steps = int(1e4), M = 100, cpus = 2, save = False, verbose = False):
    a = np.ones(N) * a_i
    L = len(a_1_array)
    Gammas_rel = np.zeros(L)
    Errors = np.zeros(L)
    
    for sigma in sigmas:
        for m, a1 in enumerate(a_1_array):
            if verbose: print(f"\nRunning for sigma={sigma} & share parameter {a1}...")
            a[0] = a1

            stats_array = simulate(N, x_ini, a, mu, sigma, steps = steps, M = M, cpus = cpus, verbose = True)
            Gammas_coop = np.array([get_growth(stats["X_coop"])[-1] for stats in stats_array])
            Gammas_def  = np.array([get_growth(stats["X_def"]) [-1] for stats in stats_array])

            _, _ , gamma_rel, error = gamma_stats(Gammas_coop, Gammas_def, agent_id=0)
            Gammas_rel[m] = gamma_rel
            Errors[m] = error
            rel_error = error/gamma_rel

            if save:
                if verbose: print("Saving results...")
                import csv
                row = [N, x_ini, a_i, a1, mu, sigma, steps, M, gamma_rel, error, rel_error]
                with open('./data/fig_1a.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)
    return


"""
function get_dynamics

Thyis function determines the dynamic that the system follows throughout its evolution.
"""

def get_dynamics(a, exponent, agent_id = 0):
    if len(a.shape) == 2: steps, N = a.shape; M = 1
    if len(a.shape) == 3: M, steps, N, = a.shape
    monopoly_count = 0
    commune_count = 0
    for m in range(M):
        # Monopoly check
        monopoly = np.logical_or( np.isclose(a[m, :, agent_id,], np.ones(steps)), np.isclose(a[m, :, agent_id], np.zeros(steps)) )
        monopoly_count += np.sum(monopoly)/steps
        # Communal check
        I = np.ones(steps)
        commune = np.isclose(a[m, :, agent_id], (I-I/N)**exponent, rtol=0.10)
        commune_count += np.sum(commune)/steps


    monopoly_count /= M
    commune_count /= M

    dynamic = "unknown"
    if   monopoly_count > 0.90: dynamic = "monopoly"
    elif commune_count  > 0.90: dynamic = "commune"
    elif (monopoly_count + commune_count  > 0.90): dynamic = "mixed"

    dynamics = {
        "monopoly": monopoly_count,
        "commune": commune_count,
        "dynamic": dynamic,
        "N_simulations": M,
    }

    return dynamics

"""
function: get_critical_exponent

This function attempts to find the greedy regime exponent where the behavior of the system changes
"""
def get_critical_exponent(N, exponent_array, sigma = 0.1, steps = int(1e4), M=10, cpus = 2, save = False, verbose = True):
    x = 10000.0
    a = 0.5
    mu = 1.0

    for nu in exponent_array:
        if verbose: print(f"\nRunning for exponent={nu}...")
        stats_array = simulate(N, x, a, mu, sigma, steps = steps, M = M, cpus = cpus, verbose = True, new_a=('greedy', nu))

        a_matrix = np.array([stats["a_array"] for stats in stats_array])
        dynamics = get_dynamics(a_matrix, nu)

        if save:
            import csv
            row = [N, sigma, steps, dynamics["N_simulations"], nu,
                   dynamics["monopoly"],dynamics["commune"],dynamics["dynamic"]]
            with open('./data/behavioral_a.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
    return

if __name__=="__main__":

    # Get crital exponent of greedy mode
    if True:
        exponent_array = np.around(np.arange(
            1, 10, 0.5        ), 3)
        get_critical_exponent(2, exponent_array, sigma = 0.1, steps = int(1e4), M = 20, cpus = 7, save = True, verbose=True)

    # Fig 1a
    if False:
        N=2
        x_ini=100.0
        a_i = 0.5
        a_1_array = np.around(np.arange(0, 1.5, 0.02)[1:], 4)
        sigmas = [0.1, 0.075, 0.050, 0.025]
        fig_1a_simulation(N, x_ini, a_i, a_1_array, 1.0, sigmas = sigmas,
                        steps=int(1e4), M = 100, cpus = 7, save = True, verbose=True)
        
    # Fig 1 suplementary
    if False:
        N_array = [2,3,4,6,10]
        a_1_array = np.around(np.arange(0, 1.5, 0.02)[1:], 4)

        # N_array = [2]
        # a_1_array = [1.4]

        fig_1_simulation(N_array, a_1_array, M=100, steps =int(1e4), save = True, cpus= 7, verbose=True)





    if False:
        A = np.array([
            [1,1,1,1,1,1,1,1,0,1],
            [1,1,1,1,1,1,1,1,0,0],
            [1,1,1,1,1,1,1,1,0,0],
            [1,1,1,1,1,1,1,1,0,0],
            [1,1,1,1,1,1,1,1,0,0],
            [1,1,1,1,1,1,1,1,0,0],
            [1,1,1,1,1,1,1,1,0,0],
            [1,1,1,1,1,1,1,1,0,0],

            [0,0,0,0,0,0,0,0,1,1],
            [1,0,0,0,0,0,0,0,1,1],
        ])
        A = np.array([
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1]
        ])

        A = np.array([
            [1, 0, 0, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 1, 0, 0, 1]
        ])

        A = np.array([
            [1,0,0,0,1,1,1,0],
            [0,0,0,1,1,1,0,0],
            [1,1,1,1,0,0,0,0],
            [0,0,0,0,1,1,0,1],
            [1,0,1,0,1,0,1,0],
            [1,1,0,1,1,0,1,1],
            [0,0,1,0,0,0,1,0],
            [1,0,1,1,1,0,1,0],
        ])

        A = np.array([
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0],
            [0,0,0,0,1,1,1,1],
            [0,0,0,0,1,1,1,1],
            [0,0,0,0,1,1,1,1],
            [0,0,0,0,1,1,1,1],
        ])



        N = len(A)
        x = np.ones(N)
        a = np.ones(N)*0.1

        _, _, Gamma_coop, Gamma_def = simulate(x, a, M= 50, steps = int(1e4), verbose=True, Adj = A)

        import plot
        plot.graph(A)
        plot.growth_rates(Gamma_coop, Gammas_def=Gamma_def)
        input()


