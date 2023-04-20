import numpy as np
import torch
import torch.nn.functional as Func

import quantum_rbm_research.utils as utils

# Add 2D RBM with sampling and verification
# make function that maps transverse ising weights to classical using
# formulas
# use paper and equation to convert BM to RBM by turning bonds into hidden neurons


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

        try:
            if W.size() != self.W.size():
                raise ValueError('dimension error')
            self.W = W
        except ValueError as err:
            print("set_weights error: " + repr(err))

    def set_vis_bias(self, vis_bias):
        """
        Description: Set visible bias tensor
        Parameters:
            vis_bias (Tensor): input vis biases
        """

        try:
            if vis_bias.size() != self.vis_bias.size():
                raise ValueError('dimension error')
            self.vis_bias = vis_bias
        except ValueError as err:
            print("set_vis_bias error: " + repr(err))

    def set_hid_bias(self, hid_bias):
        """
        Description: Set hidden bias tensor
        Parameters:
            hid_bias (Tensor): input hid biases
        """

        try:
            if hid_bias.size() != self.hid_bias.size():
                raise ValueError('dimension error')
            self.hid_bias = hid_bias
        except ValueError as err:
            print("set_hid_bias error: " + repr(err))

    def prob_h_given_v(self, vis):
        """
        Description: Calculate p(h==1|v)
        Parameters:
            vis (Tensor): visible node values
        Returns:
            prob_h (Tensor): conditional prob. of hidden nodes
        """

        prob_h = torch.sigmoid(Func.linear(vis, self.W.t(), self.hid_bias))
        return prob_h

    def prob_v_given_h(self, hid):
        """
        Description: Calculate p(v==1|h)
        Parameters:
            hid (Tensor): hidden node values
        Returns:
            prob_v (Tensor): conditional prob. of visible nodes
        """

        prob_v = torch.sigmoid(Func.linear(hid, self.W, self.vis_bias))
        return prob_v

    def sample_h(self, vis):
        """
        Description: Sample hidden nodes given visible nodes using given
        sampling distribution
        Parameters:
            vis (Tensor): visible node values
        Returns:
            hid_bin (Tensor): sampled hidden values from chosen distribution
        """

        prob_h = self.prob_h_given_v(vis)
        hid_bin = torch.bernoulli(prob_h)

        return hid_bin

    def sample_v(self, hid):
        """
        Description: Sample hidden nodes given visible nodes using given
        sampling distribution
        Parameters:
            hid (Tensor): hidden node values
        Returns:
            vis_bin (Tensor): sampled hidden values from chosen distribution
        """

        prob_v = self.prob_v_given_h(hid)
        vis_bin = torch.bernoulli(prob_v)

        return vis_bin

    def gibbs_step(self, vis):
        """
        Description: Perform a gibbs step given visible values
        Parameters:
            vis (Tensor): visible values
        Returns:
            sampled_vis (Tensor): sampled visible values
        """

        sampled_hid = self.sample_h(vis)
        sampled_vis = self.sample_v(sampled_hid)
        return sampled_vis, sampled_hid

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

        energy = -(v.t() @ self.W @ h
                   + v.t() @ self.vis_bias
                   + h.t() @ self.hid_bias
                   )
        return energy

    def get_boltzmann_distribution(self):
        """
        Description: get boltzmann distribution of RBM.
        Returns:
            boltzmann_dist (Tensor): (2^N x N+1) Tensor, where
                boltzmann_dist[:, :self.num_vis] = vis node configuration
                boltzmann_dist[:, self.num_vis:-1] = hid node configuration
                boltzmann_dist[:, -1] = probability
        """

        N = self.num_hid + self.num_vis
        boltzmann_dist = torch.cat(
            (utils.permutations(N), torch.zeros((2**N, 1))),
            dim=1
        )

        # Calculate all energy configurations in batch
        # vis (2^N x num_vis): each row is visible config
        # hid (2^N x num_hid): each row is hidden config
        # weight energy:
        #  sum((vis@W)*hid, dim=1) (2^N x 1) = column where each row represents
        #  weight energy v.t()@W@h
        # bias energy:
        #   vis@vis_bias (2^N x 1) = column where each row represents bias
        #   energy (same for hidden)
        vis = boltzmann_dist[:, :self.num_vis]
        hid = boltzmann_dist[:, self.num_vis:-1]

        energy = -(torch.sum(
            vis @ self.W * hid, dim=1)
            + vis @ self.vis_bias.t()
            + hid @ self.hid_bias.t()
        )

        boltzmann_dist[:, -1] = torch.exp(-energy)
        partition = torch.sum(boltzmann_dist[:, -1])
        boltzmann_dist[:, -1] /= partition

        return boltzmann_dist

    def sample_gibbs(self, vis_initial, steps=10):
        """
        Description: Sample visible and hidden nodes (discrete) by gibbs
        sampling
        Parameters:
            vis_initial (Tensor): initial visible node states
            steps (int): number of gibbs steps to take
        Returns:
            gibbs_sample (Tensor): discrete values of RBM
        """

        for i in range(steps):
            # Take gibbs step
            if i == 0:
                v, h = self.gibbs_step(vis_initial)
            else:
                v, h = self.gibbs_step(v)

        gibbs_sample = torch.cat((v, h))
        return gibbs_sample

    def get_gibbs_distribution(self, steps=10, samples=10000):
        """
        Description: run sample_gibbs [samples] times to generate a
        distribution
        Parmeters:
            vis_initial (Tensor): initial visible node states
            steps (int): number of gibbs steps to take
            samples (int): number of times to run sample_gibbs
        Returns:
            gibbs_dist (Tensor): (see get_boltzmann_distribution
            return description)
        """

        N = self.num_vis + self.num_hid
        gibbs_dist = torch.cat(
            (utils.permutations(N), torch.zeros((2**N, 1))),
            dim=1
        )

        # run samples, last column is count
        for i in range(samples):
            vis_init = torch.randint(1, (self.num_vis,)).float()
            sample = self.sample_gibbs(vis_init, steps=steps)
            mask = (gibbs_dist[:, :N] == sample).all(dim=1)
            gibbs_dist[mask, -1] += 1

        # last column count -> proportion
        gibbs_dist[:, -1] /= samples

        return gibbs_dist

    def learn(self, input_data):
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
        self.W += momentum * (pos_stat - neg_stat)

        # Update bias
        self.vis_bias += momentum * torch.sum(input_data - vis_prob, dim=0)
        self.hid_bias += momentum * torch.sum(pos_hid_prob - hid_prob, dim=0)

        # Compute and report squared error
        self.error = torch.sum((input_data - vis_prob)**2)
        return self.error


class RBM2D():
    def __init__(self, vis, hid, WH, WV, N, n, obc=False):
        """
        This is a class for an RBM with 2D visible and hidden nodes.
        Attributes:
            vis (2d Tensor): hidden nodes
            hid (3d Tensor): visible nodes
        """

        self.vis = vis
        self.hid = hid

        self.WH_L = 0.5 * np.arccosh(np.exp(2 * np.abs(WH)))
        self.WH_R = np.sign(WH) * self.WH_L

        self.WV_T = 0.5 * np.arccosh(np.exp(2 * np.abs(WV)))
        self.WV_B = np.sign(WV) * self.WV_T

        self.N = N
        self.n = n

        self.obc = obc

    def sample_hid(self):
        # Matrix for hidden node probability
        hid_prob = torch.zeros_like(self.hid)
        # Calculating horizontal hidden nodes
        # Add periodicity

        first_col = torch.reshape(self.vis[0, :, 0], (1, self.n, 1))
        vis_periodic = torch.cat((self.vis, first_col), dim=2)

        # Reshape to fit conv1d function
        vis_periodic = torch.reshape(vis_periodic, (self.n, 1, self.N + 1))

        # Horizontal filter
        filt_h = torch.Tensor([[[self.WH_L, self.WH_R]]])

        # Convolve
        hidden_horizontal = Func.conv1d(vis_periodic, filt_h)

        # Reshape back to fit and set hidden[0] to horizontal hidden
        hidden_horizontal = torch.reshape(hidden_horizontal, self.hid[0].shape)
        hid_prob[0] = hidden_horizontal

        # Calculate vertical hidden nodes
        # Vertical filter
        filt_v = torch.Tensor([[[[self.WV_T], [self.WV_B]]]])

        # Convolve
        hidden_vertical = Func.conv2d(self.vis, filt_v)

        # Pad a 0, since horizontal nodes are taller than vertical nodes
        hidden_vertical = torch.cat(
            (hidden_vertical, torch.zeros(1, 1, self.N)),
            dim=1
        )

        # Set hidden[1] to vertical hidden
        hid_prob[1] = hidden_vertical

        # Activation function and bernoulli
        self.vis = (torch.bernoulli(torch.sigmoid(hid_prob)) - 0.5) * 2
        print(self.vis)
# eqn. 21
