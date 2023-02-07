import torch
import torch.nn.functional as Func

class RBM():
    def __init__(self, num_vis, num_hid, weights=None, vis_biases=None, hid_biases=None):
        # RBM Params
        self.num_vis = num_vis
        self.num_hid = num_hid

        # Set weights & biases
        if weights:
            self.weights = weights
        else:
            self.weights = torch.randn(num_vis, num_hid)

        if vis_biases:
            self.vis_biases = vis_biases
        else:
            self.vis_biases = torch.ones(num_vis) * 0.5

        if hid_biases:
            self.hid_biases = hid_biases
        else:
            self.hid_biases = torch.zeros(hid_biase)

    def sample_hidden(self, vis_activations, activation='sigmoid'):
        # Choose activation function
        if activation=='sigmoid':
            act_func = self._sigmoid
        elif activation='tanh':
            act_func = self._tanh
        else:
            return "invalid activation function"

        # Perform sampling
        hid_activations = act_func(Func.linear(vis_discrete, self.weights, self.hid_biases))
        hid_discrete = torch.bernoulli(hid_activations)

        return hid_activations, hid_discrete

    def sample_visible(self, hid_discrete, activation='sigmoid'):
        # Choose activation function
        if activation=='sigmoid':
            act_func = self._sigmoid
        elif activation='tanh':
            act_func = self._tanh
        else:
            return "invalid activation function"

        # Perform sampling
        vis_activations = act_func(Func.linear(hid_discretee, self.weights.t(), self.vis_biases))
        vis_discrete = torch.bernoulli(vis_activations)

        return vis_activations, vis_discrete

    def sample(self, vis_initial, activation='sigmoid'):
        # Perform an upwards then downwards sample
        return self.sample_visible(self.sample_hidden(vis_initial, activation), activation)

    def gibbs_sampling(self, vis_initial, iter=10):
        vis_discrete = self.sample(vis_initial)
        for i in range(iter-1):
            vis_discrete = self.sample(vis_discrete)
        return vis_discrete

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _tanh(self, x):
        return (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))
