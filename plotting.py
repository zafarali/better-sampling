import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_mean_trajectories(trajectories, ts, true_trajectory, label='inferred', ax=None):
    """
    This will plot the mean and true trajectory obtained from a stochastic process
    and a ABC sampling technique.
    :param trajectories: the trajectories returned from a Solver object
    :param ts: The time axis
    :param true_trajectory: the "realized" trajectory from the stochastic process
    :param label: what to put on the graph
    :return:
    """
    n_samples = len(trajectories)
    mean = np.mean(trajectories, axis=0)
    std = np.std(trajectories, axis=0)
    std_err = std/ np.sqrt(n_samples)
    if len(ts) < mean.shape[0]:
        mean = mean[:len(ts)]
        std = std[:len(ts)]
        std_err = std[:len(ts)]

    trajectory_length = mean.shape[0]
    dimensions = mean.shape[1]
    COLORS = sns.color_palette('colorblind', dimensions + 1)

    if ax is None:
        plt.clf()
        ax = plt.gca()

    for i in range(dimensions):
        ax.fill_between(ts, mean[:, i] + std_err[:, i], mean[:, i] - std_err[:, i], alpha=0.2, color=COLORS[i])
        ax.plot(ts, mean[:, i], '--', label=label + ' mean of $x_{}$'.format(i), color=COLORS[i])

    for i in range(dimensions):
        ax.plot(ts, true_trajectory[:, i], label='true $x_{}$'.format(i), color=COLORS[i])
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel(r"$x_i$")
    return ax


def plot_trajectory_time_evolution(trajectories, dimension=0, step=5, ax=None):
    if ax is None:
        plt.clf()
        ax = plt.gca()
    COLORS = sns.color_palette('Blues_d', n_colors=len(trajectories))
    for i in range(0, len(trajectories), step):
        ax.plot(trajectories[i][:, dimension], color=COLORS[i])

    ax.set_xlabel('Time')
    ax.set_ylabel('x_i')
    ax.set_title('Evolution of Trajectories over time\n (lighter color is more recent)')
    return ax