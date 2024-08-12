import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = "./data/relative_long_term_growth_rate.csv"  
df = pd.read_csv(file_path)
"""
In some cases we will have an experiment that was done several times.
The parameters of an experiment are:
N_agents, x_ini, a_i, a_1, mu, sigma, time_steps
N_simulations determines how "powerful" the experiment was. AKA how many simulations were performed
The results of the simulations are:
gamma_1, gamma_1_def, gamma_1_rel, error, rel_error

If two experiments have the same parameters we can combine their results to get a singular more powerful one.
For our current needs we would only need to calculate the gamma_1_rel and error of the combined experiment
"""
param_columns = ['N_agents', 'x_ini', 'a_i', 'a_1', 'mu', 'sigma', 'time_steps']

grouped = df.groupby(param_columns).apply(
    lambda x: pd.Series({
        'N_simulations': x['N_simulations'].sum(),
        'gamma_1_rel': (x['gamma_1_rel'] * x['N_simulations']).sum() / x['N_simulations'].sum(),
        'error':  np.sqrt((x['error']**2 * x['N_simulations']).sum()) /  np.sqrt(x['N_simulations'].sum()),
    })
).reset_index()


output_file_path = "./data/fig1_compiled.csv"  # Cambia esto por la ruta donde quieras guardar el archivo
grouped.to_csv(output_file_path, index=False)

print(f"Resultados combinados guardados en: {output_file_path}")



df = pd.read_csv(output_file_path)
N_agents = pd.unique(df['N_agents'])

for N in N_agents:
        experiment = df[((df['N_agents']==N))]
        x = experiment['a_1']
        y = experiment['gamma_1_rel']
        error = experiment['error']
        # plt.scatter(x,y, label=f"{N} agents")
        plt.errorbar(x,y, error, label=f"{N} agents")

        id_max= np.argmax(y)
        plt.scatter(x.iloc[id_max], y.iloc[id_max])

ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax)
plt.vlines(0.5, ymin=ymin, ymax = ymax, color = 'black',)
plt.xlabel(r"Share parameter $\alpha_1$")
plt.ylabel(r"Relative long term growth rate")
plt.legend()
plt.show()
