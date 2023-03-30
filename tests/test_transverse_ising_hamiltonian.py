import numpy as np
import torch
import unittest

from quantum_rbm_research.Hamiltonian import TransverseIsingHamiltonian
import quantum_rbm_research.utils as utils


class TestTranverseIsingHamiltonian(unittest.TestCase):
    def test_example(self):
        """
        Test that calculated Hamiltonian Matrix (2-spin OBC) matches expected
        (derivation found in slides under Exercise: 2-spin Hamiltonian)
        """
        N = 2
        J = 2
        h = 1
        obc = True
        H_model = TransverseIsingHamiltonian(N, J, h, obc=obc)
        H_calculated = H_model.H

        H_expected = torch.Tensor([
            [-J, -h, -h, 0],
            [-h, J, 0, -h],
            [-h, 0, J, -h],
            [0, -h, -h, -J]
        ])

        torch.testing.assert_close(
            H_expected,
            H_calculated,
            msg="2-spin OBC Hamiltonian Matrix incorrect"
        )

    def test_e0(self):
        """
        Test ground state energy methods for Model. Check that e0_analytic()
        matches e0_eig(), which uses np.linalg.eig() to get minimum eigenvalue
        (minimum energy)
        """
        for N in range(2, 10):
            J = np.random.randint(1, 9)
            h = np.random.randint(1, 9)
            ham = TransverseIsingHamiltonian(N, J, h)

            e0_eig = ham.e0_eig()
            e0_analytic = ham.e0_analytic()

            torch.testing.assert_close(
                e0_eig,
                e0_analytic,
                atol=0,
                rtol=.00001,
                msg=f"Eig e0: {e0_eig}\nAnalytic e0: {e0_analytic}")

    def test_gs(self):
        """
        Test ground state methods for Model. Check that gs_suzuki and gs_simp,
        which are the calculated ground state vectors using Suzuki Decomp.
        of Imaginary Time Prop and simplification of Imaginary Time Prop
        (more info in slides under Suzuki-Trotter Decomp. of Ground State
        (Trans Ising) and Example: 2-spin OBC TFI), return the same vector
        (check that simplification is correct)
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
            msg=f"Ground state suzuki: {gs_suzuki}\n"
            + f"Ground State simp: {gs_simp}"
        )


if __name__ == "__main__":
    unittest.main()
