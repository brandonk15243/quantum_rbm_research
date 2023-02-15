import numpy as np
import torch
from torch import Tensor
import unittest

from quantum_rbm_research.Models import RBM
import quantum_rbm_research.utils as utils

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

    def test_gibbs_sampling(self):
        """
        Test that gibbs sampling returns learned distribution after
        training
        """
        num_vis, num_hid = 3, 2
        test_RBM = RBM(num_vis, num_hid, k=25, alpha=2)
        epochs = 50
        target = torch.Tensor([0,1,1])

        for ep in range(epochs):
            test_RBM.cdk(target)

        check = test_RBM.gibbs_sampling(
            torch.randn(num_vis),
            samples=400
            )

        # Vis node config with highest proportion should match target
        # and store in new tensor
        vis_dist = torch.empty((2**num_vis, num_vis+1))
        # First, group by visible configs
        vis_configs = utils.combinations(num_vis)
        for i, vis_config in enumerate(vis_configs):
            # Get mask to group by
            mask = (check[:,:num_vis]==vis_config).all(dim=1)
            # Get sum of grouped
            prop = check[mask, -1].sum()
            # Remove all rows
            check = check[~mask]
            # Create new row with aggregated sum
            vis_dist[i, :num_vis] = vis_config
            vis_dist[i, -1] = prop

        # Vis node config with highest proportion should match target
        max_ind = (vis_dist[:,-1]==torch.max(vis_dist[:,-1])).nonzero()[0][0]
        torch.testing.assert_close(vis_dist[max_ind,:num_vis], target)

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

    def test_boltzmann_distribution(self):
        """
        Test that sampled distribution follows Boltzmann distribution.
        Distribution should follow
        p(v,h) = exp(-beta * E) / Z
        """

        num_vis, num_hid = 2,2
        W = torch.Tensor([[1,1],[1,1]])
        test_RBM = RBM(num_vis, num_hid)
        test_RBM.set_weights(W)

        # First, get all possible configurations
        N = num_vis + num_hid
        dist = test_RBM.get_boltzmann_distribution()
        #print(dist)




if __name__=="__main__":
    unittest.main()
