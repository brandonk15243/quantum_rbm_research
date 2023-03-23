import quantum_rbm_research.utils as utils
from quantum_rbm_research.Models import RBM
from vis_funcs import *

import matplotlib.pyplot as plt
import torch


# Generate plots comparing calculated Boltzmann distribution and sampled Gibbs
# distribution of RBM with given weights

num_vis, num_hid = 2, 2
W = torch.Tensor([
    [1, -1],
    [-1, 1]
])

RBM = RBM(num_vis, num_hid)
RBM.set_vis_bias(torch.zeros(num_vis))
RBM.set_hid_bias(torch.zeros(num_hid))
RBM.set_weights(W)

dfdist = utils.dfdist_from_RBM(RBM)

fullfig = plot_full(dfdist)

visfig = plot_groupby("vis", dfdist)

savefig(fullfig, 'fullfig')
savefig(visfig, 'visfig')

