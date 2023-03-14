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

    def test_operator_at(self):
        # Number of particles
        N = 2

        # Params
        op = utils.sigma_x()

        # Calculate spin operator at index
        spinx_at_0 = utils.operator_at(op, 0, N)
        spinx_at_1 = utils.operator_at(op, 1, N)

        # Correct answers
        spinx_at_0_answer = torch.Tensor([
            [0,0,1,0],
            [0,0,0,1],
            [1,0,0,0],
            [0,1,0,0]])

        spinx_at_1_answer = torch.Tensor([
            [0,1,0,0],
            [1,0,0,0],
            [0,0,0,1],
            [0,0,1,0]])

        torch.testing.assert_close(spinx_at_0,spinx_at_0_answer)
        torch.testing.assert_close(spinx_at_1,spinx_at_1_answer)


if __name__=="__main__":
    unittest.main()
