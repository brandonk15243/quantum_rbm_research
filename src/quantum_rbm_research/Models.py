import numpy as np
import torch
import torch.nn.functional as Func

class RBM():
    def __init__(self, num_vis, num_hid, k=1, learning_rate=1e-3, batch_size=1):
        # RBM Params
        self.num_vis = num_vis
        self.num_hid = num_hid
        self.k = k
        self.learning_rate = learning_rate
        self.error = 0
        self.batch_size = batch_size

        # Weights and bias
        self.W = torch.randn(num_vis, num_hid) * np.sqrt(0.001)
        self.vis_bias = torch.ones(num_vis) * 0.25
        self.hid_bias = torch.zeros(num_hid)

    def set_weights(self, W):
        tmp_W = torch.Tensor(W)
        if tmp_W.size()==self.W.size():
            self.W = tmp_W

    def set_vis_bias(self, vis_bias):
        tmp_vis_bias = torch.Tensor(vis_bias)
        if tmp_vis_bias.size() == self.vis_bias.size():
            self.vis_bias = tmp_vis_bias

    def set_hid_bias(self, hid_bias):
        tmp_hid_bias = torch.Tensor(hid_bias)
        if tmp_hid_bias.size() == self.hid_bias.size():
            self.hid_bias = tmp_hid_bias

    def sample_hidden(self, vis_discrete, activation='sigmoid'):
        # Choose activation function
        if activation=='sigmoid':
            act_func = self._sigmoid
        elif activation=='tanh':
            act_func = self._tanh
        else:
            return "invalid activation function"

        # Perform sampling
        hid_prob = act_func(
            Func.linear(
                vis_discrete,
                self.W.t(),
                self.hid_bias
                )
            )
        hid_bin = torch.bernoulli(hid_prob)

        return hid_prob, hid_bin

    def sample_visible(self, hid_discrete, activation='sigmoid'):
        # Choose activation function
        if activation=='sigmoid':
            act_func = self._sigmoid
        elif activation=='tanh':
            act_func = self._tanh
        else:
            return "invalid activation function"

        # Perform sampling
        vis_prob = act_func(
            Func.linear(
                input=hid_discrete,
                weight=self.W,
                bias=self.vis_bias
                )
            )
        vis_bin = torch.bernoulli(vis_prob)

        return vis_prob, vis_bin

    def sample(self, vis_initial, activation='sigmoid'):
        # Perform an upwards then downwards sample
        return self.sample_visible(
            self.sample_hidden(
                vis_initial,
                activation)[1],
            activation
            )

    def train(self, input_data):
        # First forward pass
        # Collect positive statistic <p_ip_j>_{data}
        pos_hid_prob, pos_hid_bin = self.sample_hidden(input_data)
        pos_statistic_data = torch.outer(input_data, pos_hid_prob)

        # Contrastive Divergence k-times
        # "Reconstruction"
        hid_bin = pos_hid_bin
        for i in range(self.k):
            # Use hidden binary vals when getting visible prob.
            vis_prob = self.sample_visible(hid_bin)[0]
            hid_prob, hid_bin = self.sample_hidden(vis_prob)

        # Last pass
        # Collect negative statistic <p_ip_j>_{reconstructed}
        neg_statistic_recon = torch.outer(vis_prob, hid_prob)

        # Update weights
        # (Hinton) When using mini-batches, divide by size of mini-batch
        self.W += (self.learning_rate / self.batch_size) * (pos_statistic_data - neg_statistic_recon)

        # Update bias
        self.vis_bias += (self.learning_rate / self.batch_size) * torch.sum(input_data - vis_prob, dim=0)
        self.hid_bias += (self.learning_rate / self.batch_size) * torch.sum(pos_hid_prob - hid_prob, dim=0)

        # Compute and report squared error
        self.error = torch.sum((input_data - vis_prob)**2)
        return self.error

    def gibbs_sampling(self, vis_initial, samples=10, round=False):
        distribution = torch.zeros(self.num_vis)
        vis_bin = self.sample(vis_initial)[1]
        for i in range(samples-1):
            vis_bin = self.sample(vis_bin)[1]
            distribution += vis_bin
        if round:
            return torch.round(distribution/samples)
        return distribution/samples

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _tanh(self, x):
        return (
            (torch.exp(x) - torch.exp(-x)) /
            (torch.exp(x) + torch.exp(-x))
            )

class BM():
    def __init__(self):
        return
