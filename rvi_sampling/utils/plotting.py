import torch
import torch.nn.functional as F
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

    if true_trajectory is not None:
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

def conduct_draws_nn(sp_, x, t, n_draws=100, feed_time=False):
    """
    Conducts draws from a neural network policy and takes the average.
    :param sp_: The policy
    :param x: The position in the random walk
    :param t: the time in the random walk
    :return:
    """
    with torch.set_grad_enabled(False):
        if feed_time:
            data = torch.FloatTensor([[x, t]])
        else:
            data = torch.FloatTensor([[x]])

        log_probs = sp_.fn_approximator(data)
        probs = F.softmax(log_probs, dim=1).data # convert log probs to probs

    # return the expected step:
    return torch.mean(probs * torch.FloatTensor([-1, 1]))

def conduct_draws(sp_, x, t, n_draws=100):
    """
    Conducts draws from a regular (non-neural ne twork) policy and takes the average
    :param sp_:
    :param x:
    :param t:
    :return:
    """

    return np.mean(sp_.draw([[x]], t, sampling_probs_only=True) * np.array([-1, 1]))


def visualize_proposal(list_of_proposals,
                       timesteps,
                       xranges,
                       neural_network=True):
    """
    Takes a list of policies and returns the components needed to visualize using plt.quiver.

    For example:
    ```
    t, x, x_arrows, y_arrows_list = visualize_proposal([polic1, policy2], 50, 10)
    plt.quiver(t, x, x_arrows, y_arrows_list[0])
    plt.quiver(t, x, x_arrows, y_arrows_list[1])
    ```
    :param list_of_proposals:
    :param timesteps:
    :param xranges:
    :return:
    """
    t, x = np.meshgrid(range(0, timesteps), range(-xranges, xranges + 1))
    vector_grid_arrows_x = np.zeros_like(t)

    vector_grid_y_arrows = [[] for _ in list_of_proposals]
    for x_ in x[:, 0]:
        vector_grid_y_arrows_t = [[] for _ in list_of_proposals]
        for t_ in t[0, :]:
            for proposal, vector_grid_y_arrows_t_i in zip(list_of_proposals, vector_grid_y_arrows_t):
                if neural_network:
                    # neural network draw must be "reversed"
                    feed_time = proposal.fn_approximator.Input.weight.size()[1] == 2
                    vector_grid_y_arrows_t_i.append(-conduct_draws_nn(proposal, float(x_), t_ / timesteps, feed_time=feed_time))
                else:
                    # hand designed proposals are already going backward in time.
                    vector_grid_y_arrows_t_i.append(conduct_draws(proposal, float(x_), t_))
        for i in range(len(vector_grid_y_arrows_t)):
            vector_grid_y_arrows[i].append(vector_grid_y_arrows_t[i])
    return t, x, vector_grid_arrows_x, vector_grid_y_arrows


def visualize_penalties(list_of_proposals,
                        timesteps,
                        xranges,
                        neural_network=True):
    """
    Takes a list of proposals and plots the penalties (i.e.
    :param list_of_proposals:
    :param timesteps:
    :param xranges:
    :param neural_network:
    :return:
    """
    raise NotImplementedError('Not implemented yet.')

def determine_panel_size(n_panels):
    """
    Determines panel size based on the number of panels we have
    :param n_panels:
    :return:
    """
    assert n_panels <= 9, 'We can only plot at most 9 panels at a time'

    if n_panels in [7, 8, 9]:
        return '33'
    elif n_panels in [5, 6]:
        return '23'
    elif n_panels in [4]:
        return '22'
    elif n_panels in [3]:
        return '13'
    elif n_panels in [2]:
        return '12'
    elif n_panels in [1]:
        return '11'

def multi_quiver_plot(t, x, x_arrows, proposal_arrows, titles=None, figsize=None):
    """
    Plots at most 9 quiver plots on one figure.
    :param t:
    :param x:
    :param x_arrows:
    :param proposal_arrows: a list of proposal visualizations
    :return:
    """

    panel_size = determine_panel_size(len(proposal_arrows))
    f = plt.figure() if figsize is None else plt.figure(figsize=figsize)

    for i, proposal in enumerate(proposal_arrows):
        ax = f.add_subplot(panel_size+str(i+1))
        ax.quiver(t, x, x_arrows, proposal)
        if titles is not None: ax.set_title(titles[i])
        ax.set_xlabel('Time')
        ax.set_ylabel('x')

    f.tight_layout()
    return f

