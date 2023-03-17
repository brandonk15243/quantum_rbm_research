import pandas as pd
import torch
from torch import Tensor
import unittest

import quantum_rbm_research.utils as utils


class TestUtils(unittest.TestCase):
    def test_permutations(self):
        # Number of bits
        N = 4
        perm = utils.permutations(N)

        # Assert appropriate length
        self.assertTrue(perm.size() == (2 ** N, N))

    def test_permutations_df(self):
        num_vis, num_hid = 3, 2
        perm = utils.permutations_df(num_vis, num_hid)

        self.assertTrue(
            perm['vis'].unique().size == 2 ** num_vis,
            msg="Unique visible permutation size incorrect"
        )

        self.assertTrue(
            perm['hid'].unique().size == 2 ** num_hid,
            msg="Unique hidden permutation size incorrect"
        )

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
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]])

        spinx_at_1_answer = torch.Tensor([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]])

        torch.testing.assert_close(spinx_at_0, spinx_at_0_answer)
        torch.testing.assert_close(spinx_at_1, spinx_at_1_answer)


if __name__ == "__main__":
    unittest.main()
