import numpy as np
import scipy as sp
import torch
from torch import Tensor
import unittest

from quantum_rbm_research.Hamiltonian import TransverseIsingHamiltonian
import quantum_rbm_research.utils as utils


class TestTranverseIsingHamiltonian(unittest.TestCase):
    def test_example(self):
        """
        Test that calculated Hamiltonain (2-spin OBC) matches expected
        """
        N = 2
        J = 2
        h = 1
        obc = True
        H_expected = torch.Tensor([
            [-J, -h, -h, 0],
            [-h, J, 0, -h],
            [-h, 0, J, -h],
            [0, -h, -h, -J]
        ])

        H_model = TransverseIsingHamiltonian(N, J, h, obc=obc)
        H_calculated = H_model.H

        torch.testing.assert_close(
            H_expected,
            H_calculated,
            msg="2-spin OBC Hamiltonian Matrix incorrect"
        )

    def test_gs(self):
        """
        Test that Suzuki Trotter decomposition returns ground state vector:
        lim_{n->infty}(e^{-Delta tau*H0}e^{-Delta tau*H1})^n|psi> propto |psi0>
        Also test that simplification of Suzuki trotter returns same state
        """
        N = 2
        J = 2
        h = 1
        obc = True
        H_model = TransverseIsingHamiltonian(N, J, h, obc=obc)

        initial_state = torch.rand(2**N)
        initial_state /= np.linalg.norm(initial_state)

        tau = 25
        n = 100000
        gs_suzuki = H_model.gs_suzuki_trotter(tau, n, initial_state)
        gs_simp = utils.twospin_e0(J, h, tau, n, initial_state)

        torch.testing.assert_close(
            gs_suzuki,
            gs_simp,
            msg="Ground state by Suzuki differs from "
            + "ground state by simplified Suzuki"
        )

    def test_e0(self):
        for N in range(10):
            J = np.random.randint(1, 9)
            h = np.random.randint(1, 9)
            ham = TransverseIsingHamiltonian(N, J, h)

            e0_eig = ham.e0_eig()
            e0_analytic = ham.e0_analytic()

            self.assertAlmostEqual(e0_eig, e0_analytic, 3)


if __name__ == "__main__":
    unittest.main()
