import numpy as np
import torch
import torch.nn.functional as Func

import quantum_rbm_research.utils as utils

class RBM():
    def __init__(self, num_vis, num_hid, k=1, alpha=1e-3, batch_size=1):
        """
        This is a class for a general RBM.
        Attributes:
            num_vis (int): number of visible nodes
            num_hid (int): number of hidden nodes
            k (int): parameter for CD-k
            alpha (float): learning rate
            batch_size (int): batch size for batched learning
        """
        # Params
        self.num_vis = num_vis
        self.num_hid = num_hid
        self.k = k
        self.alpha = alpha
        self.error = 0
        self.batch_size = batch_size

        # Weights and biases
        self.W = torch.randn(num_vis, num_hid) * np.sqrt(0.001)
        self.vis_bias = torch.ones(num_vis) * 0.25
        self.hid_bias = torch.zeros(num_hid)

    def set_weights(self, W):
        """
        Description: Set weight tensor
        Parameters:
            W (Tensor): input weight matrix
        """
        if W.size()==self.W.size():
            self.W = W

    def set_vis_bias(self, vis_bias):
        """
        Description: Set visible bias tensor
        Parameters:
            vis_bias (Tensor): input vis biases
        """
        if vis_bias.size() == self.vis_bias.size():
            self.vis_bias = vis_bias

    def set_hid_bias(self, hid_bias):
        """
        Description: Set hidden bias tensor
        Parameters:
            hid_bias (Tensor): input hid biases
        """
        if hid_bias.size() == self.hid_bias.size():
            self.hid_bias = hid_bias

    def prob_h_given_v(self, vis, activation='sigmoid'):
        """
        Description: Calculate p(h==1|v)
        Parameters:
            vis (Tensor): visible node values
            activation (str): choice of activation function
        Returns:
            prob_h (Tensor): conditional prob. of hidden nodes
        """
        if activation=='sigmoid':
            act_func = self._sigmoid
        else:
            print("Invalid activation function. Defaulting to sigmoid")
            act_func = self._sigmoid

        prob_h = act_func(Func.linear(vis, self.W.t(), self.hid_bias))
        return prob_h

    def prob_v_given_h(self, hid, activation='sigmoid'):
        """
        Description: Calculate p(v==1|h)
        Parameters:
            hid (Tensor): hidden node values
            activation (str): choice of activation function
        Returns:
            prob_v (Tensor): conditional prob. of visible nodes
        """
        if activation=='sigmoid':
            act_func = self._sigmoid
        else:
            print("Invalid activation function. Defaulting to sigmoid")
            act_func = self._sigmoid

        prob_v = act_func(Func.linear(hid, self.W, self.vis_bias))
        return prob_v

    def sample_h(self, vis, activation='sigmoid', sampling_dist='bernoulli'):
        """
        Description: Sample hidden nodes given visible nodes using given
        sampling distribution
        Parameters:
            vis (Tensor): visible node values
            activation (str): choice of activation function
            sampling_dist (str): choice of sampling distribution
        Returns:
            hid_bin (Tensor): sampled hidden values from chosen distribution
        """
        if sampling_dist=='bernoulli':
            sampling_func = torch.bernoulli
        else:
            print("Invalid sampling dist. Defaulting to bernoulli")
            sampling_func = torch.bernoulli

        prob_h = self.prob_h_given_v(vis, activation)
        hid_bin = sampling_func(prob_h)

        return hid_bin

    def sample_v(self, hid, activation='sigmoid', sampling_dist='bernoulli'):
        """
        Description: Sample hidden nodes given visible nodes using given
        sampling distribution
        Parameters:
            hid (Tensor): hidden node values
            activation (str): choice of activation function
            sampling_dist (str): choice of sampling distribution
        Returns:
            vis_bin (Tensor): sampled hidden values from chosen distribution
        """
        if sampling_dist=='bernoulli':
            sampling_func = torch.bernoulli
        else:
            print("Invalid sampling dist. Defaulting to bernoulli")
            sampling_func = torch.bernoulli

        prob_v = self.prob_v_given_h(hid, activation)
        vis_bin = sampling_func(prob_v)

        return vis_bin

    def cdk(self, input_data):
        """
        Description: Train model using contrastive divergence where
        parameter k=self.k
        Parameters:
            input_data (Tensor): data to train model towards
        Returns:
            self.error (Tensor): squared error after training
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

    def gibbs_step(self, vis, activation='sigmoid', sampling_dist='bernoulli'):
        """
        Description: Perform a gibbs step given visible values
        Parameters:
            vis (Tensor): visible values
            activation (str): choice of activation function
            sampling_dist (str): choice of sampling distribution
        Returns:
            sampled_vis (Tensor): sampled visible values
        """
        sampled_vis = self.sample_v(
            self.sample_h(vis, activation, sampling_dist),
            activation,
            sampling_dist
            )
        return sampled_vis

    def gibbs_sampling(self, vis_initial, samples=10):
        """
        Description: Perform gibbs sampling
        Parameters:
            vis_initial (Tensor): initial values to begin sampling
            samples (int): number of gibbs steps to perform
            mode (str): which nodes (visible, hidden) to return
        Returns:
            dist_tensor (Tensor): distribution (last column = proportions)
        """

        # dist_tensor: (2^N x N+1)
        #   dist_tensor[:, :N] = node configuration (visible then hidden)
        #   dist_tensor[:, -1] = proportion of samples with configuration of row
        N = self.num_vis + self.num_hid
        dist_tensor = torch.cat(
            (utils.combinations(N), torch.zeros((2**N,1))),
            dim=1
            )

        for i in range(samples):
            # Get config of vis and hid nodes
            if i==0:
                hid_bin = self.sample_h(vis_initial)
            else:
                hid_bin = self.sample_h(vis_bin)
            vis_bin = self.sample_v(hid_bin)

            # Concatenate into 1 tensor (row vector)
            config = torch.cat((vis_bin, hid_bin), dim=0)

            # Get mask of matching column
            mask = (dist_tensor[:,:N]==config).all(dim=1)

            # Add to count
            dist_tensor[mask, -1] += 1

        # Divide by sample size
        dist_tensor[:,-1] /= samples
        return dist_tensor

    def energy(self, v, h):
        """
        Description: compute the energy of the RBM given vectors v and h
        representing binary values of visible and hidden nodes
        Parameters:
            v (Tensor): binary values of visible nodes
            h (Tensor): binary values of hidden nodes
        Returns:
            energy of RBM
        """
        return -(v.t()@self.W@h + v.t()@self.vis_bias + h.t()@self.hid_bias)

    def get_boltzmann_distribution(self):
        """
        Description: get boltzmann distribution of RBM.

        Returns:
            dist (Tensor):
        """

        # dist_tensor: (2^N x N+1)
        #   dist_tensor[:, :N] = node configuration (visible then hidden)
        #   dist_tensor[:, -1] = proportion of samples with configuration of row
        N = self.num_hid + self.num_vis
        dist_tensor = torch.cat((utils.combinations(N), torch.zeros((2**N,1))), dim=1)

        # Calculate all energy configurations in batch
        # vis (2^N x num_vis): each row is visible config
        # hid (2^N x num_hid): each row is hidden config
        # weight energy:
        #  sum((vis@W)*hid, dim=1) (2^N x 1) = column where each row represents weight energy
        #   v.t()@W@h
        # bias energy:
        #   vis@vis_bias (2^N x 1) = column where each row represents bias energy
        #   (same for hidden)
        vis = dist_tensor[:,:self.num_vis]
        hid = dist_tensor[:,self.num_vis:-1]

        energy = -(
            torch.sum(vis@self.W*hid,dim=1) +
            vis@self.vis_bias.t() +
            hid@self.hid_bias.t()
            )

        dist_tensor[:,-1] = torch.exp(-energy)
        partition = torch.sum(dist_tensor[:,-1])
        dist_tensor[:,-1] /= partition

        return dist_tensor

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _tanh(self, x):
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


# eqn. 21
