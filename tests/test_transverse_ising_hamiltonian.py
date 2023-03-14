import numpy as np
import torch
from torch import Tensor
import unittest

from quantum_rbm_research.Hamiltonian import TransverseIsingHamiltonian
import quantum_rbm_research.utils as utils

class TestTranverseIsingHamiltonian(unittest.TestCase):
    def test_H(self):
        for N in range(10):
            J = 2
            h = 1
            ham = TransverseIsingHamiltonian(N, J, h)

            ground_state_eig = np.min(np.linalg.eig(ham.H)[0])
            ground_state_analytic = utils.tfi_e0(N, J, h)
            self.assertAlmostEqual(ground_state_eig,ground_state_analytic, 3)

# compare ground state energy

if __name__=="__main__":
    unittest.main()
