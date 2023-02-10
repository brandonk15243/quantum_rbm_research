import numpy as np
import torch
from torch import Tensor
import unittest

import quantum_rbm_research.utils as utils

class TestUtils(unittest.TestCase):
    def test_combinations(self):
        # Number of bits
        N = 4
        comb = utils.combinations(N)

        # Assert appropriate length
        self.assertTrue(comb.size()==(2**N,N))



if __name__=="__main__":
    unittest.main()
