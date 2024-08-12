import numpy as np    
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



def update_a(x):
    X_total = np.sum(x)
    X_rel = x/X_total
    a = 1 - X_rel # Greedy mode
    a = 1 - 1/X_rel # Generosity mode
    # print(X_rel)
    return a

def _evolve_defection(N, x, a, mu, sigma, steps):
    time = range(1, steps+1) 
    x_def = np.copy(x)
    x_ini = np.copy(x)

    X_coop = np.zeros((steps+1, N))
    X_def = np.zeros((steps+1, N))
    X_coop[0], X_def[0] = x_ini, x_ini
    for i in time:
        a = update_a(x)
        dseta = np.random.normal(mu, sigma)
        x = x * dseta*(1 - a) + np.mean(a*x*dseta)
        x_def = x_def*dseta

        X_coop[i] = x
        X_def[i] = x_def
            
    return X_coop, X_def

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



if __name__=="__main__":

    x = np.linspace(1,10, 10)
    X, X_def = evolve(10, x, sigma = 0.075)
    import plot
    plot.evolution(X)