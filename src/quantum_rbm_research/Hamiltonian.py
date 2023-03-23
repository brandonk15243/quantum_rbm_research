import quantum_rbm_research.utils as utils

import numpy as np
from scipy.linalg import expm
import torch

"""
Transverse Ising Hamiltonian with open boundary conditions
# add periodic
"""


class TransverseIsingHamiltonian():
    def __init__(self, N, J, h, obc=False):
        # Params
        self.N = N
        self.J = J
        self.h = h
        self.obc = obc

        # Create matrix
        self.H = self._H0() + self._H1()

    def e0_analytic(self):
        """
        Description: calculate (analytically) the ground state energy for the
        given N, J, h (from Quantum Ising Phase Transitions in Transverse Ising
        Models by Chakrabarti, with my own edits)
        Parameters:
            N (int): number of spins
            J (int): interaction term (does this have to be an int?)
            h (int): transverse field term (does this have to be an int?)
        Returns:
            e0 (float): ground state energy
        """
        m = np.arange(-(self.N - 1) / 2, (self.N + 1) / 2, 1)
        q = 2 * np.pi * m / self.N

        # lambda bar (pg. 13)
        lam = self.J / self.h

        # w_q (pg. 20)
        omega_q = np.sqrt(1 + 2 * lam * np.cos(q) + lam**2)

        # I noticed that for h > J, the returned value was off by a factor of
        # h from the minimum eigenvalue, which is why I added the second term.
        e0 = -np.sum(omega_q) * np.max([self.h, 1])
        return e0

    def e0_eig(self):
        """
        Description: Get ground state energy of self.H using minimum
        eigenvalue and eigenvector
        Returns:
            e0 (Tensor): ground state energy
        """
        return np.min(np.linalg.eig(self.H)[0])

    def gs_eig(self):
        """
        Description: Get ground state vector of self.H using minimum
        eigenvalue and eigenvector
        Returns:
            psi0 (Tensor): ground state vector
        """
        return np.linalg.eig(self.H)[1][:, 0]

    def gs_suzuki_trotter(self, tau, n, initial_state=None):
        """
        Description: Get ground state vector of self.H using Suzuki
        Trotter decompoisition and High Energy Filtering through Imaginary
        Time Propagation
        Parameters:
            tau (float): Time propagation
            n (int): num. Trotter steps
        Returns:
            psi0 (Tensor): ground state vector (should match eigenvector
            associated with min eigenvalue)
        """
        if initial_state is not None:
            psi0 = initial_state
        else:
            psi0 = torch.rand(2**self.N)
            psi0 /= np.linalg.norm(psi0)
        delta_tau = tau / n

        # Suzuki Trotter decomp of imaginary time propagation
        time_prop = (
            torch.Tensor(expm(-delta_tau * self._H0()))
            @ torch.Tensor(expm(-delta_tau * self._H1()))
        )

        # Operator n times
        for i in range(n):
            psi0 = time_prop @ psi0
            if n % 50 == 0:
                psi0 /= np.linalg.norm(psi0)

        return psi0

    def _H0(self):
        """
        Description: Return Hamiltonian matrix containing just interaction
        terms (J)
        Returns:
            H0 (Tensor): 2^N x 2^N matrix given by sum of adjacent spin-z
            operators
        """
        H0 = torch.zeros((2**self.N, 2**self.N))
        N = (self.N - 1) if self.obc else self.N
        for i in range(0, N):
            # Interaction term
            j = (i + 1) % self.N
            spinz_i = utils.operator_at(utils.sigma_z(), i, self.N)
            spinz_j = utils.operator_at(utils.sigma_z(), j, self.N)
            H0 += spinz_i @ spinz_j

        return -self.J * H0

    def _H1(self):
        """
        Description: Return Hamiltonian matrix containing just external field
        terms (h)
        Returns:
            H1 (Tensor): 2^N x 2^N matrix given by sum of adjacent spin-z
            operators
        """
        H1 = torch.zeros((2**self.N, 2**self.N))
        for i in range(0, self.N):
            # External field term
            spinx_i = utils.operator_at(utils.sigma_x(), i, self.N)
            H1 += spinx_i

        return -self.h * H1
