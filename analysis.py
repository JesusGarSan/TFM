import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# df = pd.read_csv("./data/relative_long_term_growth_rate_no_threshold.csv")
df = pd.read_csv("./data/relative_long_term_growth_rate.csv")
df = df.sort_values(by=['N_agents', 'a_1'], ascending=False)
N_agents = pd.unique(df['N_agents'])

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

# def filter_df(df, parameters):
#         filtered_df = df
#         for key, value in parameters.items():
#                 print(key, value)
#                 filtered_df = filtered_df[filtered_df[key]==value]
#         return filtered_df



# parameters = {
#         "N_agents": 2,
#         "x_ini" : 1,
#         "a_i" : 0.5,
#         "a_1" : 0.5,
#         "mu" : 1.0,
#         "sigma" : 0.1,
#         "time_steps" : 10000,
# }
# df = filter_df(df, parameters)
# print(df)
# quit()
for N in N_agents:
        experiment = df[((df['N_agents']==N) & (df['N_simulations']==500))]
        print(experiment)
        x = experiment['a_1']
        y = experiment['gamma_1_rel']
        error = experiment['error']

        # plt.scatter(x,y, label=f"{N} agents")
        plt.errorbar(x,y, error, label=f"{N} agents")

ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax)
plt.vlines(0.5, ymin=ymin, ymax = ymax, color = 'black',)
plt.xlabel(r"Share parameter $\alpha_1$")
plt.ylabel(r"Relative long term growth rate")
plt.legend()
plt.show()
