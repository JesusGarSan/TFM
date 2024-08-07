from agent_model import *
def simulate(x_ini,a,mu,sigma, threshold = 0.0, steps = int(1e4) ,M=1,plot=False,verbose=False):
    assert len(x_ini) == len(a), "Número de agentes inconsistente \n Inconsistent number of agents"
    N = len(x_ini)
    X, X_def, Gammas, Gammas_def = np.zeros((N,M)),np.zeros((N,M)),np.zeros((N,M)),np.zeros((N,M))

    # Simulations
    for m in range(M):
        if plot:
            x, x_def, gamma, gamma_def = evolve(x_ini, a, mu, sigma, threshold, steps = steps, save_steps=True)
            plot_evolution(x)
            plot_evolution(x_def)
            x,x_def,gamma,gamma_def = x[-1],x_def[-1],gamma[-1],gamma_def[-1]
        if not plot:
            x, x_def, gamma, gamma_def = evolve(x_ini, a, mu, sigma, threshold, steps = steps, save_steps=False)
        X[:,m], X_def[:,m], Gammas[:,m], Gammas_def[:,m] = x,x_def,gamma,gamma_def
    
    return X, X_def, Gammas, Gammas_def


def run(N_array, a_1_array, M = 10, steps=int(1e4), x_ini=1.0, a_i=0.5, mu = 1.0, sigma=0.1, threshold=0.0, save = False, verbose = False):
    for N in N_array:
        x = np.ones(N) * x_ini
        for a_1 in a_1_array:
            if verbose: print(f"Running for {N} agents & share parameter {round(a_1, 2)}...")
            a = np.ones(N) * a_i
            a[0] = a_1

            _, _, Gammas, Gammas_def = simulate(x,a,mu,sigma, threshold,steps,M)

            Gammas_rel = (Gammas[0,:] - Gammas_def[0,:]) * 100
            gamma = np.mean(Gammas[0,:]) * 100
            gamma_def = np.mean(Gammas_def[0,:]) * 100
            gamma_rel = np.mean(Gammas_rel)
            error = np.std(Gammas_rel)/np.sqrt(len(Gammas_rel))
            rel_error = error/gamma_rel

            if save:
                import csv
                row = [N, x_ini, a_i, a_1, mu, sigma, threshold, steps, M, gamma, gamma_def, gamma_rel, error, rel_error]
                with open('./data/relative_long_term_growth_rate.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)



if __name__=="__main__":
    pass

    N = [2,3,4,6,10]
    a = np.arange(0, 1.5, 0.02)[1:]
    run(N, a, M = 20, save=True, verbose=True)