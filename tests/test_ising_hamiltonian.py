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
        tau = 0.1
        n = 5
        TFIModel = TransverseIsingHamiltonian(N, J, h, obc=obc)

        IModel = TFIModel.convert_to_classical(tau, n)

        fig = vis_funcs.plot_ising(IModel)
        fig.show()
        plt.show()

        delta_tau = tau / n

        self.assertEqual(IModel.WH, delta_tau * J)
        self.assertEqual(IModel.WV, -0.5 * np.log(np.tanh(delta_tau * h)))

    def test_to_rbm(self):
        N = 4
        J = 2
        h = 1
        obc = False
        TFIModel = TransverseIsingHamiltonian(N, J, h, obc=obc)

        IModel = TFIModel.convert_to_classical(0.1, 10)

        rbm = IModel.convert_to_rbm2d()


if __name__ == "__main__":
    unittest.main()
