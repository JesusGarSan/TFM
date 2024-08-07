"""
Vamos a crear redes con diferentes topologías
de tal modo que los agentes colindantes interactúen entre sí.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



adj_matrix = np.array([
    [0,1,1,1,1,1,1],
    [1,0,1,0,0,0,0],
    [1,1,0,1,0,0,0],
    [1,0,1,0,0,0,0],
    [1,0,0,0,0,1,0],
    [1,0,0,0,1,0,0],
    [1,0,0,0,0,0,0],
])

G = nx.from_numpy_array(adj_matrix)

# Dibujar el grafo
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', node_size=700, edge_color='gray', font_weight='bold')

# Mostrar el grafo
plt.show()


from agent_model import *

