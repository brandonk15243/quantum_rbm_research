import quantum_rbm_research.utils as utils

import numpy as np
import torch

"""
Transverse Ising Hamiltonian with open boundary conditions
"""

class TransverseIsingHamiltonian():
    def __init__(self, N, J, h):
        # Params
        self.N = N
        self. J = J
        self.h = h

        # Create matrix
        self.H = torch.zeros(2**self.N,2**self.N)
        for i in range(1,self.N+1):
            # Interaction term
            if i < self.N:
                spinz_i = utils.operator_at(utils.sigma_z(), i, self.N)
                spinz_j = utils.operator_at(utils.sigma_z(), i+1, self.N)
                self.H += -self.J * spinz_i@spinz_j

            # External field term
            spinx_i = utils.operator_at(utils.sigma_x(), i, self.N)

            self.H += -self.h * spinx_i
