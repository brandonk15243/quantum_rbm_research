import quantum_rbm_research.utils as utils
from quantum_rbm_research.Models import RBM

import matplotlib.pyplot as plt
import numpy as np
import time
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

    def test_gibbs_sample(self):
        """
        Test that gibbs sampling diverges to expected distribution
        """
        num_vis, num_hid = 2, 2
        test_RBM = RBM(num_vis, num_hid, k=20, alpha=2)
        epochs = 10
        target = torch.Tensor([1,1])
        for ep in range(epochs):
            test_RBM.learn(target)

        init = torch.randn(num_vis)

        test_RBM.gibbs_sample_probability(init, steps=60000)


    def test_energy(self):
        """
        Test that energy function works as intended on simple model
        """
        num_vis, num_hid = 2, 2
        test_W = torch.Tensor([[1,-1],[0,1]])
        test_vis_bias = torch.Tensor([1,2])
        test_hid_bias = torch.Tensor([1,0])
        vis, hid = torch.randn(num_vis), torch.randn(num_hid)

        test_RBM = RBM(num_vis, num_hid)
        test_RBM.set_weights(test_W)
        test_RBM.set_vis_bias(test_vis_bias)
        test_RBM.set_hid_bias(test_hid_bias)

        # Energy by hand
        calc_energy = -(
            test_W[0][0]*vis[0]*hid[0] + test_W[1][0]*vis[1]*hid[0] +
            test_W[0][1]*vis[0]*hid[1] + test_W[1][1]*vis[1]*hid[1] +
            vis.t()@test_vis_bias + hid.t()@test_hid_bias
            )

        # Assert close
        RBM_energy = test_RBM.energy(vis, hid)

        torch.testing.assert_close(calc_energy, RBM_energy)

    def test_get_boltzmann_distribution(self):
        """
        Test that the get_boltzmann_distribution function
        returns the correct value (Boltzmann distribution)
        Distribution should follow
        p(v,h) = exp(-beta * E) / Z
        """

        num_vis, num_hid = 2,2
        W = torch.Tensor([[1,1],[1,1]])
        zero = torch.zeros(2)
        test_RBM = RBM(num_vis, num_hid)
        test_RBM.set_weights(W)
        test_RBM.set_vis_bias(zero)
        test_RBM.set_hid_bias(zero)

        # Calculate actual distribution
        N = num_vis + num_hid
        actual_dist_tensor = torch.cat((utils.combinations(N),torch.zeros((2**N,1))), dim=1)
        for row in actual_dist_tensor:
            # calculate energy
            row[-1] = test_RBM.energy(row[:num_vis],row[num_vis:-1])
        # take exponential since p(v,h)=exp(-E(v,h))
        actual_dist_tensor[:, -1] = torch.exp(-actual_dist_tensor[:,-1])
        # divide by partition
        actual_dist_tensor[:, -1] /= torch.sum(actual_dist_tensor[:,-1])

        # get calculated dist
        calculated_dist_tensor = test_RBM.get_boltzmann_distribution()

        # check equality
        torch.testing.assert_close(actual_dist_tensor, calculated_dist_tensor)

    def test_distribution_gibbs(self):
        """
        Test that sampled (by gibbs) distribution follows the boltzmann distribution
        """
        num_vis, num_hid = 2,2
        W = torch.Tensor([[2,3],[0,1]])
        zero = torch.zeros(2)
        test_RBM = RBM(num_vis, num_hid)
        test_RBM.set_weights(W)
        test_RBM.set_vis_bias(zero)
        test_RBM.set_hid_bias(zero)

        boltzmann = test_RBM.get_boltzmann_distribution()

        N = num_vis+num_hid
        sampled_dist = torch.cat((utils.combinations(N),torch.zeros((2**N,1))), dim=1)

        # get sampled distribution
        samples = 10000
        for _ in range(samples):
            rand = torch.randint(1,(num_vis,)).float()
            sampled = test_RBM.sample_gibbs(rand, steps=5)
            mask = (sampled_dist[:,:N]==sampled).all(dim=1)
            sampled_dist[mask,-1] += 1

        sampled_dist[:,-1] /= samples

        print(sampled_dist[:,-1]-boltzmann[:,-1])

        torch.testing.assert_close(boltzmann[:,-1],sampled_dist[:,-1])


if __name__=="__main__":
    unittest.main()
