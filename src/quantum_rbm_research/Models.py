import numpy as np
import torch
import torch.nn.functional as Func

class RBM():
    def __init__(self, num_vis, num_hid, k=1, alpha=1e-3, batch_size=1):
        """
        this is a class for an RBM

        Attributes:
            num_vis (int): number of visible nodes
            num_hid (int): number of hidden nodes
            k (int): parameter for CD-k
            alpha (float): learning rate
            batch_size (int): batch size for batched learning
        """
        # RBM Params
        self.num_vis = num_vis
        self.num_hid = num_hid
        self.k = k
        self.alpha = alpha
        self.error = 0
        self.batch_size = batch_size

        # Weights and bias
        self.W = torch.randn(num_vis, num_hid) * np.sqrt(0.001)
        self.vis_bias = torch.ones(num_vis) * 0.25
        self.hid_bias = torch.zeros(num_hid)

    def set_weights(self, W: torch.Tensor):
        """
        Definition: set RBM weights
        Parameters:
            W (Tensor): input weight matrix
        """
        if W.size()==self.W.size():
            self.W = W

    def set_vis_bias(self, vis_bias: torch.Tensor):
        """
        Description: set RBM visible biases
        Parameters:
            vis_bias (Tensor): input vis biases
        """
        if vis_bias.size() == self.vis_bias.size():
            self.vis_bias = vis_bias

    def set_hid_bias(self, hid_bias: torch.Tensor):
        """
        Description: Set RBM hidden biases
        Parameters:
            hid_bias (Tensor): input hid biases
        """
        if hid_bias.size() == self.hid_bias.size():
            self.hid_bias = hid_bias

    def prob_h_given_v(self, vis: torch.Tensor, activation='sigmoid'):
        """
        Description: get conditional prob. of h given v values
        Parameters:
            vis (Tensor): visible node values
            activation (str): choice of activation function
        Returns:
            prob_h: conditional prob. of hidden nodes
        """
        if activation=='sigmoid':
            act_func = self._sigmoid
        else:
            print("Invalid activation function. Defaulting to sigmoid")
            act_func = self._sigmoid

        prob_h = act_func(Func.linear(vis, self.W.t(), self.hid_bias))
        return prob_h

    def sample_h(
        self,
        vis: torch.Tensor,
        activation='sigmoid',
        sampling_dist='bernoulli'):
        """
        Description: sample h nodes using chosen sampling distribution
        from conditional probability
        Parameters:
            vis (Tensor): visible node values
            activation (str): choice of activation function
            sampling_dist (str): choice of sampling distribution
        Returns:
            hid_bin: sampled hidden values from chosen distribution
        """
        if sampling_dist=='bernoulli':
            sampling_func = torch.bernoulli
        else:
            print("Invalid sampling dist. Defaulting to bernoulli")
            sampling_func = torch.bernoulli

        prob_h = self.prob_h_given_v(vis, activation)

        hid_bin = sampling_func(prob_h)

        return hid_bin

    def prob_v_given_h(self, hid: torch.Tensor, activation='sigmoid'):
        """
        Description: get conditional prob. of v given h values
        Parameters:
            hid (Tensor): hidden node values
            activation (str): choice of activation function
        Returns:
            prob_v: conditional prob. of visible nodes
        """
        if activation=='sigmoid':
            act_func = self._sigmoid
        else:
            print("Invalid activation function. Defaulting to sigmoid")
            act_func = self._sigmoid

        prob_v = act_func(Func.linear(hid, self.W, self.vis_bias))
        return prob_v

    def sample_v(
        self,
        hid: torch.Tensor,
        activation='sigmoid',
        sampling_dist='bernoulli'):
        """
        Description: sample v nodes using chosen sampling distribution
        from conditional probability
        Parameters:
            hid (Tensor): hidden node values
            activation (str): choice of activation function
            sampling_dist (str): choice of sampling distribution
        Returns:
            vis_bin: sampled hidden values from chosen distribution
        """
        if sampling_dist=='bernoulli':
            sampling_func = torch.bernoulli
        else:
            print("Invalid sampling dist. Defaulting to bernoulli")
            sampling_func = torch.bernoulli

        prob_v = self.prob_v_given_h(hid, activation)

        vis_bin = sampling_func(prob_v)

        return vis_bin

    def gibbs_step(self, vis: torch.Tensor, activation='sigmoid',
        sampling_dist='bernoulli'):
        """
        Description: perform a gibbs step given visible values
        Parameters:
            vis (Tensor): visible values
            activation (str): choice of activation function
            sampling_dist (str): choice of sampling distribution
        Returns:
            sampled visible values
        """
        # Perform a gibbs step given visible values
        return self.sample_v(
            self.sample_h(vis, activation, sampling_dist),
            activation,
            sampling_dist
            )

    def cdk(self, input_data: torch.Tensor):
        """
        Description: train model using contrastive divergence method
        Parameters:
            input_data (Tensor): data to train model towards
        Returns:
            self.error: squared error after training
        """
        # First forward pass
        # Collect positive statistic <p_ip_j>_{data}
        pos_hid_prob = self.prob_h_given_v(input_data)
        pos_hid_bin = self.sample_h(input_data)
        pos_stat = torch.outer(input_data, pos_hid_prob)

        # Contrastive Divergence k-times
        # "Reconstruction"
        hid_bin = pos_hid_bin
        for i in range(self.k):
            # Use hidden binary vals when getting visible prob.
            vis_prob = self.prob_v_given_h(hid_bin)
            hid_bin = self.sample_h(vis_prob)

        # Last pass
        # Collect negative statistic <p_ip_j>_{reconstructed}
        hid_prob = self.prob_h_given_v(vis_prob)
        neg_stat = torch.outer(vis_prob, hid_prob)

        # Update weights
        # (Hinton) When using mini-batches, divide by size of mini-batch
        momentum = self.alpha / self.batch_size
        self.W += momentum * (pos_stat- neg_stat)

        # Update bias
        self.vis_bias += momentum * torch.sum(input_data - vis_prob, dim=0)
        self.hid_bias += momentum * torch.sum(pos_hid_prob - hid_prob, dim=0)

        # Compute and report squared error
        self.error = torch.sum((input_data - vis_prob)**2)
        return self.error

    def gibbs_sampling(self, vis_initial: torch.Tensor, samples=10):
        """
        Description: perform gibbs sampling given initial visible values
        [samples] times
        Parameters:
            vis_initial (Tensor): initial values to begin sampling
            samples (int): number of gibbs steps to perform
        Returns:
            histogram of distribution
        """
        distribution = torch.zeros(self.num_vis)
        vis_bin = self.gibbs_step(vis_initial)
        for i in range(samples-1):
            vis_bin = self.gibbs_step(vis_bin)
            distribution += vis_bin
        return distribution/samples

    def energy(self, v: torch.Tensor, h: torch.Tensor):
        """
        Description: compute the energy of the RBM given vectors v and h
        representing binary values of visible and hidden nodes
        Parameters:
            v (Tensor): binary values of visible nodes
            h (Tensor): binary values of hidden nodes
        Returns:
            energy of RBM
        """
        return -(v.t()@self.W@h.t() + v.t()@self.vis_bias + h.t()@self.hid_bias)

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

# eqn. 21
