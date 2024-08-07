import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./data/relative_long_term_growth_rate.csv")

N_agents = pd.unique(df['N_agents'])


for N in N_agents:
        experiment = df[((df['N_agents']==N) & (df['N_simulations']==100))]
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
