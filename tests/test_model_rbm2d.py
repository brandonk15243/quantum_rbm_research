import quantum_rbm_research.utils as utils
from quantum_rbm_research.Models import RBM, RBM2D
from quantum_rbm_research.Hamiltonian import TransverseIsingHamiltonian

import numpy as np
import torch
from torch import Tensor
import unittest


class TestModelRBM2D(unittest.TestCase):
    def test(self):
        N = 2
        J = 2
        h = 1
        n = 3
        obc = True

        trans_ising_model = TransverseIsingHamiltonian(N, J, h, obc=obc)

        ising_model = trans_ising_model.convert_self_to_classical(0.5, n)

        rbm2d = ising_model.convert_self_to_rbm2d()

        vis_init_3dim = (torch.bernoulli(torch.empty((1, n, N)).uniform_(0, 1)) - 0.5) * 2
        prob_h = rbm2d.prob_h_given_v(vis_init_3dim)

        hid_init_3dim = (torch.bernoulli(torch.empty((2, n, N)).uniform_(0, 1)) - 0.5) * 2
        prob_v = rbm2d.prob_v_given_h(hid_init_3dim)


        dist = rbm2d.get_gibbs_distribution(steps=3, samples=100)
        print(dist)


if __name__ == "__main__":
    unittest.main()
