import numpy as np
import torch
from torch import Tensor
import unittest

from quantum_rbm_research.Hamiltonian import TransverseIsingHamiltonian
import quantum_rbm_research.utils as utils

class TestTranverseIsingHamiltonian(unittest.TestCase):
    def test_H(self):
        ham = TransverseIsingHamiltonian(2, 1, 0)
        print(ham.H)

if __name__=="__main__":
    unittest.main()
