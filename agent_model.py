import numpy as np

# Crecimiento multiplicativo estocástico
def evolve(x, a, mu, sigma, threshold=None, steps = 1000, save_steps = False):
    assert len(x) == len(a), "Número de agentes inconsistente \n Inconsistent number of agents"
    N = len(x)

    time = range(steps) 
    x_def = np.copy(x)
    x_ini = np.copy(x)

    if save_steps:
        X = np.zeros((steps, N))
        X_def = np.zeros((steps, N))
        Gamma = np.zeros((steps, N))
        Gamma_def = np.zeros((steps, N))
        for i in time:
            dseta = np.random.normal(mu, sigma, N)
            x = x * dseta*(1-a) + np.mean(a*x*dseta)
            x_def = x_def*dseta

            X[i] = x
            X_def[i] = x_def
            Gamma[i] = np.log(x/x_ini)/(i+1)
            Gamma_def[i] =  np.log(x_def/x_ini)/(i+1)
            
        return X, X_def, Gamma, Gamma_def

    if not save_steps:
        for i in time:
            dseta = np.random.normal(mu, sigma, N)
            x = x * dseta*(1-a) + np.mean(a*x*dseta)
            x_def = x_def*dseta
            
        gamma = np.log(x/x_ini)/steps
        gamma_def =  np.log(x_def/x_ini)/steps
        return x, x_def, gamma, gamma_def
    

    
# Añadimos un threshold. Si la población cae demasiado ya no será capaz de seguir adelante
def evolve_beta(x, a, mu, sigma, threshold = 0.0, steps = 1000, save_steps = False):
    assert len(x) == len(a), "Número de agentes inconsistente \n Inconsistent number of agents"
    N = len(x)

    time = range(steps) 
    x_def = np.copy(x)
    x_ini = np.copy(x)

    if save_steps:
        X = np.zeros((steps, N))
        X_def = np.zeros((steps, N))
        Gamma = np.zeros((steps, N))
        Gamma_def = np.zeros((steps, N))
        for i in time:
            dseta = np.random.normal(mu, sigma, N)
            idx = np.nonzero(x)
            idx_def = np.nonzero(x_def)
            n = len(idx)
            x[idx] = x[idx] * dseta[idx]*(1- 2/n *a[idx]) + np.mean(a[idx]*x[idx]*dseta[idx])
            x_def[idx_def] = x_def[idx_def]*dseta[idx_def]
            
            # Eliminamos los agentes que no sobrepasen el threshold
            x[np.where(x < threshold )] = 0
            x_def[np.where(x_def < threshold )] = 0


            X[i] = x
            X_def[i] = x_def
            Gamma[i] = np.nan_to_num( np.log(x/x_ini)/(i+1) )
            Gamma_def[i] =  np.nan_to_num( np.log(x_def/x_ini)/(i+1) )
            
        return X, X_def, Gamma, Gamma_def

    if not save_steps:
        for i in time:
            dseta = np.random.normal(mu, sigma, N)
            idx = np.nonzero(x)
            idx_def = np.nonzero(x_def)
            n = len(idx)
            x[idx] = x[idx] * dseta[idx]*(1- 2/n *a[idx]) + np.mean(a[idx]*x[idx]*dseta[idx])
            x_def[idx_def] = x_def[idx_def]*dseta[idx_def]

            # Eliminamos los agentes que no sobrepasen el threshold
            x[np.where(x < threshold )] = 0
            x_def[np.where(x_def < threshold )] = 0

        gamma = np.nan_to_num( np.log(x/x_ini)/steps )
        gamma_def =  np.nan_to_num( np.log(x_def/x_ini)/steps )

        return x, x_def, gamma, gamma_def


def plot_evolution(x, legend = True):
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







if __name__=="__main__":
    pass

    # # Generación de datos para recrear la figura suplementaria 1
    # N = [2,3,4,6,10]
    # a = np.arange(0, 1.5, 0.01)[1:]
    # run(N, a, M = 100, threshold=0.1, save=True, verbose=True)

    # default
    N = 2
    x0 = 1
    a_i = 0.2
    x_ini = np.ones(N)*x0
    a = np.ones(N) * a_i
    steps = int(1e3)
    mu = 1
    sigma = 0.1  
    threshold= 0.1

    
    