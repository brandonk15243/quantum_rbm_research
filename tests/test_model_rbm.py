import numpy as np
import torch
from torch import Tensor
import unittest

from quantum_rbm_research.Models import RBM

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
        num_vis, num_hid = 4, 2
        test_RBM = RBM(num_vis, num_hid, k=25, alpha=2)
        epochs = 50
        target = torch.Tensor([0,1,1,0])

        for ep in range(epochs):
            test_RBM.cdk(target)

        check = torch.round(
            test_RBM.gibbs_sampling(
                torch.randn(num_vis),
                samples=400
                )
            )

        torch.testing.assert_close(
            target,
            check
            )


if __name__=="__main__":
    unittest.main()
