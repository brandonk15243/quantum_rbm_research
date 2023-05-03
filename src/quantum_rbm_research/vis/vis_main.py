import quantum_rbm_research.utils as utils
from quantum_rbm_research.Models import RBM
from quantum_rbm_research.Hamiltonian import TransverseIsingHamiltonian
from vis_funcs import *

import matplotlib.pyplot as plt
import torch


# Generate plots comparing calculated Boltzmann distribution and sampled Gibbs
# distribution of RBM with given weights

num_vis, num_hid = 2, 2
W = torch.Tensor([
    [1, 1],
    [1, 1]
])
vis_bias = torch.Tensor([1, 1])
hid_bias = torch.Tensor([1, 1])

RBM = RBM(num_vis, num_hid)
RBM.set_vis_bias(vis_bias)
RBM.set_hid_bias(hid_bias)
RBM.set_weights(W)

dfdist = utils.dfdist_from_RBM(RBM)

fullfig = plot_full(dfdist)

visfig = plot_groupby("vis", dfdist)


savefig(fullfig, 'fullfig_updated')
savefig(visfig, 'visfig_updated')

# N = 2
# J = 2
# h = 1
# obc = True
# ham = TransverseIsingHamiltonian(N, J, h, obc)
# errfig = plot_gs_eig_trotter_error(ham)

# savefig(errfig, 'errfig')
