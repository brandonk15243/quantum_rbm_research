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
        test_weights = np.random.rand(num_vis, num_hid)
        test_vis_biases = torch.randn(num_vis)
        test_hid_biases = [1 for _ in range(num_hid)]

        test_RBM = RBM(num_vis, num_hid)

        # Set weights and biases
        test_RBM.set_weights(test_weights)
        test_RBM.set_vis_biases(test_vis_biases)
        test_RBM.set_hid_biases(test_hid_biases)

        # Check equality
        self.assertNotIn(
            False,
            torch.eq(Tensor(test_weights), test_RBM.weights),
            msg="Model weights set incorrectly"
            )

        self.assertNotIn(
            False,
            torch.eq(Tensor(test_vis_biases), test_RBM.vis_biases),
            msg="Model visible biases set incorrectly"
            )

        self.assertNotIn(
            False,
            torch.eq(Tensor(test_hid_biases), test_RBM.hid_biases),
            msg="Model hidden biases set incorrectly"
            )

    def test_gibbs_sampling(self):
        """
        Test that gibbs sampling returns learned distribution after
        training
        """
        num_vis, num_hid = 4, 2
        test_RBM = RBM(num_vis, num_hid, k=25, learning_rate=2)
        epochs = 50
        target = torch.Tensor([0,1,1,0])

        for _ in range(epochs):
            print(test_RBM.train(target)) if _%10==0 else None

        print(test_RBM.gibbs_sampling(torch.randn(num_vis), samples=400))

if __name__=="__main__":
    unittest.main()
