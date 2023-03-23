import matplotlib.pyplot as plt
import numpy as np
import os

mylocation = os.path.dirname(__file__)
figs_folder = os.path.join(mylocation, "figs")


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


def savefig(fig, title):
    fig.savefig(os.path.join(figs_folder, f'{title}.png'))
