import quantum_rbm_research.utils as utils

import numpy as np
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
        self.H = self._interaction_matr() + self._external_matr()

    def _interaction_matr(self):
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

    def _external_matr(self):
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
