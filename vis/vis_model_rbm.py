import quantum_rbm_research.utils as utils
from quantum_rbm_research.Models import RBM
from vis_funcs import *

import matplotlib.pyplot as plt
import os
import torch

cwd = os.getcwd()
figs_folder = os.path.join(cwd, "vis_model_rbm_figs")

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

boltzmann = RBM.get_boltzmann_distribution()
gibbs = RBM.get_gibbs_distribution()

dfdist = utils.permutations_df(num_vis, num_hid)
dfdist['boltz'] = boltzmann[:, -1]
dfdist['gibbs'] = gibbs[:, -1]

# Generate bar plot
figfull, axfull = plt.subplots()
dfdist.plot.bar('full', ['boltz', 'gibbs'], ax=axfull, zorder=3)

# Rotate xtick labels
axfull.set_xticklabels(dfdist['full'], rotation=45)

# Titles and Labels
axfull.set_title("Boltzmann vs. Gibbs Distribution (Full)")
axfull.set_xlabel("State")
axfull.set_ylabel("Probability")

axfull.grid()

# Save Fig
figfull.set_size_inches(10, 6)
figfull.set_layout_engine('tight')
# figfull.savefig(os.path.join(figs_folder, 'dist_comparison_full.png'))

# Grouped by visual nodes
plot_groupby_vis(dfdist)
