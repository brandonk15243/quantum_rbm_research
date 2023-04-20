import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
import numpy as np
import os
import torch

mylocation = os.path.dirname(__file__)
figs_folder = os.path.join(mylocation, "figs")

########################################
# Plot RBM graphs
########################################

def plot_full(dfdist):
    """
    Description: Given a dfdist, plot Boltzmann and Gibbs probabilities
    Parameters:
        dfdist (DataFrame): DataFrame of distribution
    Returns:
        fig (Figure): fig object with plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    dfdist.plot.bar('full', ['boltz', 'gibbs'], ax=ax, zorder=3)

    ax.set_xticklabels(dfdist['full'], rotation=45)

    ax.set_title(f"Boltzmann vs. Gibbs Distribution")
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")

    ax.grid()
    fig.set_layout_engine('tight')

    return fig


def plot_groupby(option, dfdist):
    """
    Description: Given a dfdist, plot Boltzmann and Gibbs probabilities
    after grouping by either vis or hid configurations
    Parameters:
        option (str): what to group by ('vis' or 'hid')
        dfdist (DataFrame): DataFrame of distribution
    Returns:
        fig (Figure): fig object with plot
    """
    if option not in ('vis', 'hid'):
        print("Invalid option. Choose 'vis' or 'hid'")
        return
    grouped = dfdist[[option, 'boltz', 'gibbs']].groupby(option).agg(sum)
    fig, ax = plt.subplots(figsize=(10, 6))

    grouped.plot.bar(y=['boltz', 'gibbs'], ax=ax, zorder=3)

    ax.set_xticklabels(grouped.index, rotation=45)

    ax.set_title(f"Boltzmann vs. Gibbs Distribution (Grouped by {option})")
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")

    ax.grid()
    fig.set_layout_engine('tight')

    return fig

########################################
# Plot TFI Hamiltonian graphs
########################################


def plot_gs_eig_trotter_error(ham):
    # Generate heatmap comparing n (trotter number) and tau (time step)

    # Values of n and tau
    size = 50
    n_vals = np.linspace(10, 150, size, dtype=int)
    tau_vals = np.linspace(1, 25, size)

    init_state = torch.Tensor([.8, .2, .45, .3])
    init_state /= torch.linalg.norm(init_state)

    # Error matrix
    error_matrix = np.empty((size, size))

    # Fill matrix
    for i, n in enumerate(n_vals):
        for j, tau in enumerate(tau_vals):
            gs_trotter = ham.gs_suzuki_trotter(tau, n, init_state)
            gs_eig = ham.gs_eig()
            error_matrix[i][j] = np.linalg.norm(gs_trotter - gs_eig)

    # norm_arr = error_matrix / np.linalg.norm(error_matrix)

    fig, ax = plt.subplots(figsize=(10, 6))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    im = ax.imshow(error_matrix)
    ax.invert_yaxis()

    tick_freq = 5

    ax.set_xticks(
        ticks=np.arange(size)[::tick_freq],
        labels=np.round(tau_vals, 3)[::tick_freq],
        rotation=45
    )
    ax.set_xlabel(r"$\tau$")

    ax.set_yticks(
        ticks=np.arange(size)[::tick_freq],
        labels=n_vals[::tick_freq]
    )
    ax.set_ylabel("n")

    ax.set_title(r"Error ($\tau$ vs. n)")

    fig.colorbar(im, cax=cax)
    fig.set_layout_engine('tight')

    return fig

########################################
# Plot Classical Ising graphs
########################################


def plot_ising(isingModel):

    fig, ax = plt.subplots(figsize=(10, 15))

    pos = dict((n, n) for n in isingModel.graph.nodes())

    # Draw nodes
    nx.draw_networkx_nodes(
        isingModel.graph,
        pos,
        node_size=600,
        ax=ax
    )
    # Labels
    nx.draw_networkx_labels(
        isingModel.graph,
        pos,
        font_size=9,
        font_color='w',
        ax=ax,
    )

    # Draw edges
    nx.draw_networkx_edges(
        isingModel.graph,
        pos,
        edgelist=isingModel.graph.edges(),
        width=6,
        alpha=0.5,
        edge_color='k',
        ax=ax
    )
    # Labels
    edge_labels = nx.get_edge_attributes(isingModel.graph, 'weight')
    for key in edge_labels.keys():
        edge_labels[key] = round(edge_labels[key], 3)
    nx.draw_networkx_edge_labels(
        isingModel.graph,
        pos,
        edge_labels,
        ax=ax
    )

    return fig


def savefig(fig, title):
    fig.savefig(os.path.join(figs_folder, f'{title}.png'))
