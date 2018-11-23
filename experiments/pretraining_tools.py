"""
Tools to help setup pretraining scripts.
"""

import numpy as np
import torch
import torch.nn.functional as F



def conduct_draws(proposal, x, t):
    """
    Conduct draws from the proposal distribution at position x and time t.

    Args:
    :param proposal: A proposal_distributions.ProposalDistribution object that
        can be sampled from.
    :param x: The spatial position.
    :param t: The temporal position.
    :return: A sample from the proposal.
    """
    return np.flip(proposal.draw([[x]], t, sampling_probs_only=True), 0)

def generate_data(proposal, timesteps, xranges):
    """
    Generate data to train a proposal distribution with.

    :param proposal: The proposal distribution to sample from.
    :param timesteps: The number of time steps for the stochastic process.
    :param xranges: The size of the spatial component to train on.
    :return: A tuple containing training inputs and training outputs.
    """
    t, x = np.meshgrid(range(0, timesteps), range(-xranges, xranges + 1))

    training_inputs = []
    training_outputs = []
    for x_ in x[:, 0]:
        for t_ in t[0, :]:
            training_inputs.append((float(x_), t_/timesteps))
            training_outputs.append(conduct_draws(proposal, float(x_), t_))

    training_inputs = torch.from_numpy(
        np.array(training_inputs)).view(-1, 2).float()
    training_outputs = torch.from_numpy(
        np.array(training_outputs)).view(-1, 2).float()

    return training_inputs, training_outputs

def SoftCE(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580/5
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """

    if size_average:
        return torch.mean(
            torch.sum(-target * F.log_softmax(input), dim=1))
    else:
        return torch.sum(
            torch.sum(-target * F.log_softmax(input), dim=1))