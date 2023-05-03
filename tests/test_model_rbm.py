import quantum_rbm_research.utils as utils
from quantum_rbm_research.Models import RBM

import numpy as np
import torch
from torch import Tensor
import unittest


class TestModelRBM(unittest.TestCase):
    def test_model_init(self):
        """
        Test that model initialization works as intended
        """
        num_vis, num_hid = 2, 3
        test_weights = torch.randn(num_vis, num_hid)
        test_vis_bias = torch.randn(num_vis)
        test_hid_bias = torch.randn(num_hid)

        test_RBM = RBM(num_vis, num_hid)

        # Set weights and biases
        test_RBM.set_weights(test_weights)
        test_RBM.set_vis_bias(test_vis_bias)
        test_RBM.set_hid_bias(test_hid_bias)

        # Check equality
        self.assertNotIn(
            False,
            torch.eq(Tensor(test_weights), test_RBM.W),
            msg="Model weights set incorrectly"
        )

        self.assertNotIn(
            False,
            torch.eq(Tensor(test_vis_bias), test_RBM.vis_bias),
            msg="Model visible biases set incorrectly"
        )

        self.assertNotIn(
            False,
            torch.eq(Tensor(test_hid_bias), test_RBM.hid_bias),
            msg="Model hidden biases set incorrectly"
        )

    def test_energy(self):
        """
        Test that energy function works as intended on simple model
        """
        num_vis, num_hid = 2, 2
        test_W = torch.Tensor([[1, -1], [0, 1]])
        test_vis_bias = torch.Tensor([1, 2])
        test_hid_bias = torch.Tensor([1, 0])
        vis = (torch.bernoulli(torch.empty(num_vis).uniform_(0, 1)) - 0.5) * 2
        hid = (torch.bernoulli(torch.empty(num_hid).uniform_(0, 1)) - 0.5) * 2

        test_RBM = RBM(num_vis, num_hid)
        test_RBM.set_weights(test_W)
        test_RBM.set_vis_bias(test_vis_bias)
        test_RBM.set_hid_bias(test_hid_bias)

        # Energy by hand
        expected_energy = -(
            test_W[0][0] * vis[0] * hid[0]
            + test_W[1][0] * vis[1] * hid[0]
            + test_W[0][1] * vis[0] * hid[1]
            + test_W[1][1] * vis[1] * hid[1]
            + vis.t() @ test_vis_bias + hid.t() @ test_hid_bias
        )

        # Assert close
        RBM_energy = test_RBM.energy(vis, hid)

        torch.testing.assert_close(
            expected_energy,
            RBM_energy,
            msg="Energy not close"
        )

    def test_get_boltzmann_distribution(self):
        """
        Test that the get_boltzmann_distribution function
        returns the correct value (Boltzmann distribution)
        Distribution should follow
        p(v,h) = exp(-beta * E) / Z
        """

        num_vis, num_hid = 2, 2
        W = torch.Tensor([[1, 1], [1, 1]])
        zero = torch.zeros(2)
        test_RBM = RBM(num_vis, num_hid)
        test_RBM.set_weights(W)
        test_RBM.set_vis_bias(zero)
        test_RBM.set_hid_bias(zero)

        # Calculate actual distribution
        N = num_vis + num_hid
        actual_dist_tensor = torch.cat(
            (utils.permutations_pm_one(N), torch.zeros((2**N, 1))),
            dim=1
        )

        for row in actual_dist_tensor:
            # calculate energy
            row[-1] = test_RBM.energy(row[:num_vis], row[num_vis:-1])
        # take exponential since p(v,h)=exp(-E(v,h))
        actual_dist_tensor[:, -1] = torch.exp(-actual_dist_tensor[:, -1])
        # divide by partition
        actual_dist_tensor[:, -1] /= torch.sum(actual_dist_tensor[:, -1])

        # get calculated dist
        calculated_dist_tensor = test_RBM.get_boltzmann_distribution()

        # check equality
        torch.testing.assert_close(
            actual_dist_tensor,
            calculated_dist_tensor,
            msg="Boltzmann distribution not close"
        )

    def test_get_gibbs_distribution(self):
        """
        Test that sampled (by gibbs) distribution follows the boltzmann
        distribution
        """
        num_vis, num_hid = 2, 2
        W = torch.Tensor([[2, -3], [-9, 3]])
        zero = torch.zeros(2)
        test_RBM = RBM(num_vis, num_hid)
        test_RBM.set_weights(W)
        test_RBM.set_vis_bias(zero)
        test_RBM.set_hid_bias(zero)

        boltzmann = test_RBM.get_boltzmann_distribution()
        gibbs = test_RBM.get_gibbs_distribution(steps=3, samples=10000)

        torch.testing.assert_close(
            boltzmann[:, -1],
            gibbs[:, -1],
            atol=1,
            rtol=1,
            msg="Gibbs sample distribution not close to Boltzmann"
        )


if __name__ == "__main__":
    unittest.main()
