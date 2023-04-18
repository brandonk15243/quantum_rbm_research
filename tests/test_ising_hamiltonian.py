import numpy as np
import torch
import unittest

import matplotlib.pyplot as plt
import networkx as nx

from quantum_rbm_research.Hamiltonian import TransverseIsingHamiltonian, IsingHamiltonian
import quantum_rbm_research.utils as utils


class TestIsingHamiltonian(unittest.TestCase):
    def test_conversion(self):
        N = 2
        J = 2
        h = 1
        obc = False
        TFIModel = TransverseIsingHamiltonian(N, J, h, obc=obc)

        IModel = TFIModel.convert_to_classical(0.1, 4)
        pos = dict((n, n) for n in IModel.graph.nodes())

        IModel.graph.nodes[(0,0)]['spin'] = -1

        IModel.update_weights()
        G = IModel.graph
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=700)

        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
        nx.draw_networkx_edges(
            G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
        )

        # node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    unittest.main()
