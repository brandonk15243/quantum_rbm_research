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

    def test_suzuki_trotter_simplification(self):
        """
        Test that Suzuki Trotter decomposition returns ground state vector:
        lim_{n->infty}(e^{-Delta tau*H0}e^{-Delta tau*H1})^n|psi> propto |psi0>
        Also test that simplification of Suzuki trotter returns same state
        """
        ########################################
        # Suzuki Trotter
        ########################################
        K = 20
        N = 2
        J = 2
        h = 1
        obc = True
        H_model = TransverseIsingHamiltonian(N, J, h, obc=obc)

        print("Eigen: ", np.linalg.eig(H_model.H))

        # Get non-commutain Hamiltonian parts
        H0 = H_model._interaction_matr()
        H1 = H_model._external_matr()

        # Suzuki Trotter
        # Initial state (arbitrary)
        initial_state = torch.rand(2**N)
        # Normalize
        initial_state /= np.linalg.norm(initial_state)
        print("Initial state: \n", initial_state)

        # Parameters
        # heatmap of tau and n plotting error
        tau = 1
        n = 50000
        delta_tau = tau / n

        # Suzuki Trotter
        suzuki_trotter = (
            torch.Tensor(sp.linalg.expm(-delta_tau * H0))
            @ torch.Tensor(sp.linalg.expm(-delta_tau * H1))
        )

        # Operator n times
        # see how normalization impacts error
        gs_suzuki = initial_state
        for i in range(n):
            gs_suzuki = suzuki_trotter @ gs_suzuki
            gs_suzuki /= np.linalg.norm(gs_suzuki)

        ########################################
        # Simplified
        ########################################
        gs_simp = utils.twospin_e0(J, h, tau, n, initial_state)

        print(
            "Ground state from suzuki (before simplification): \n",
            gs_suzuki
        )
        print(
            "Ground state after simplification): \n",
            gs_simp
        )

        print("Error: ", np.linalg.norm(gs_simp - np.linalg.eig(H_model.H)[1][:, 0]))

    def test_H_rand(self):
        for N in range(10):
            J = np.random.randint(1, 9)
            h = np.random.randint(1, 9)
            ham = TransverseIsingHamiltonian(N, J, h)

            ground_state_eig = np.min(np.linalg.eig(ham.H)[0])
            ground_state_analytic = utils.tfi_e0(N, J, h)
            self.assertAlmostEqual(ground_state_eig, ground_state_analytic, 3)


if __name__ == "__main__":
    unittest.main()
