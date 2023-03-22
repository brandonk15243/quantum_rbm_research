import matplotlib.pyplot as plt

def plot_groupby_vis(dfdist):
    grouped = dfdist.groupby('vis').agg(sum)
    fig, ax = plt.subplots((10,6))

    grouped.plot.bar('vis', ['boltz', 'gibbs'], ax=ax, zorder=3)

    ax.set_xticklabels(dfdist['vis'], rotation=45)

    ax.set_title("Boltzmann vs. Gibbs Distribution (Grouped by Vis)")
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")

    ax.grid()
    fig.set_layout_engine('tight')

    fig.show()
    plt.show()

    return fig
