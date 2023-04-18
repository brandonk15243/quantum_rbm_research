import numpy as np
import torch
import unittest

import matplotlib.pyplot as plt
import networkx as nx

from quantum_rbm_research.Hamiltonian import TransverseIsingHamiltonian, IsingHamiltonian
import quantum_rbm_research.utils as utils

import quantum_rbm_research.vis.vis_funcs as vis_funcs


class TestIsingHamiltonian(unittest.TestCase):
    def test_conversion(self):
        N = 4
        J = 2
        h = 1
        obc = False
        TFIModel = TransverseIsingHamiltonian(N, J, h, obc=obc)

        IModel = TFIModel.convert_to_classical(0.1, 10)

        IModel.update_weights()

        fig = vis_funcs.plot_ising(IModel)
        fig.show()
        plt.show()


if __name__ == "__main__":
    unittest.main()
