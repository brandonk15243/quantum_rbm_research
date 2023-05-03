from quantum_rbm_research.Hamiltonian import TransverseIsingHamiltonian
import quantum_rbm_research.utils as utils

import numpy as np
import torch
import unittest
import matplotlib.pyplot as plt



class TestTranverseIsingHamiltonian(unittest.TestCase):
    def test_initialization(self):
        """
        Test initialization (no z-field yet)
        """
        N = 2
        J = 2
        h_x = 1
        obc = True
        H_model = TransverseIsingHamiltonian(N, J, h_x, obc=obc)
        H_calculated = H_model.H

        H_expected = torch.Tensor([
            [-J, -h_x, -h_x, 0],
            [-h_x, J, 0, -h_x],
            [-h_x, 0, J, -h_x],
            [0, -h_x, -h_x, -J]
        ])

        torch.testing.assert_close(
            H_expected,
            H_calculated,
            msg="2-spin OBC Hamiltonian Matrix incorrect"
        )

    def test_commutation(self):
        """
        Confirm [H0, H1] do not commute
        """
        N = 2
        J = 2
        h_x = 1
        obc = True
        H_model = TransverseIsingHamiltonian(N, J, h_x, obc=obc)

        commutation = (
            H_model._H0() @ H_model._H1()
            - H_model._H1() - H_model._H0()
        )

        self.assertFalse((commutation == torch.zeros_like(commutation)).all())

    def test_e0(self):
        """
        Test ground state methods with random parameters
        """
        for N in range(2, 10):
            J = np.random.randint(1, 9)
            h_x = np.random.randint(1, 9)
            ham = TransverseIsingHamiltonian(N, J, h_x)

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
        h_x = 1
        obc = True
        H_model = TransverseIsingHamiltonian(N, J, h_x, obc=obc)

        initial_state = torch.rand(2**N)
        initial_state /= np.linalg.norm(initial_state)

        tau = 25
        n = 100000
        gs_suzuki = H_model.gs_suzuki_trotter(tau, n, initial_state)
        gs_simp = utils.twospin_e0(J, h_x, tau, n, initial_state)

        torch.testing.assert_close(
            gs_suzuki,
            gs_simp,
            msg=f"Ground state suzuki: {gs_suzuki}\n"
            + f"Ground State simp: {gs_simp}"
        )

    def test_avg_z_analytic(self):
        """
        Test average z spin for Hamiltonian
        """
        # Simple model, 2 spins, no external fields, average should be 0
        # since expectation for each spin is 0
        H_simple = TransverseIsingHamiltonian(
            2,
            1,
            0,
            obc=True
        )

        self.assertEqual(H_simple.avg_z_analytic(), 0)

        # Another simple, but with a z field included. Average should be 1
        H_simple2 = TransverseIsingHamiltonian(
            2,
            1,
            0,
            1,
            obc=True
        )

        self.assertEqual(H_simple2.avg_z_analytic(), 1)

        # More complex model, 8 spins, transverse x field with complementary
        # z field (to break symmetry)
        # Sweep h_x (and h_z) and plot results
        avg_spins = []
        avg_spins_calc = []
        h_x_norm = []
        hmax = 5
        for h_x in np.linspace(0.1, hmax, 10, dtype=float):
            # Analytical average z-spin
            J = 2.
            h_z = 1
            model = TransverseIsingHamiltonian(
                2,
                J,
                h_x,
                h_z=h_z,
                obc=False
            )
            avg_spins.append(model.avg_z_analytic(beta=10.))
            h_x_norm.append(h_x / J)

            # Sampled average z-spin
            tau = 0.5
            n = 4
            rbm2d = model.convert_self_to_classical(tau, n).convert_self_to_rbm2d()
            dist = rbm2d.get_gibbs_average_z_mag(steps=8, samples=400)
            avg_spins_calc.append(torch.mean(torch.abs(dist)))

        plt.plot(h_x_norm, avg_spins)
        plt.plot(h_x_norm, avg_spins_calc)
        plt.show()


if __name__ == "__main__":
    unittest.main()
