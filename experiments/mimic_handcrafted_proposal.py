"""
Pre-train the RVI proposal to mimic a proposal of choice.
"""

import numpy as np
import torch
import torch.nn as nn
from pg_methods.networks import MLP_factory
from torch.utils.data import TensorDataset, DataLoader
from rvi_sampling.distributions.proposal_distributions import FunnelProposal
from pg_methods.policies import CategoricalPolicy
import argparse

def conduct_draws(proposal, x, t):
    return np.flip(proposal.draw([[x]], t, sampling_probs_only=True), 0)

def generate_data(proposal, timesteps, xranges):
    t, x = np.meshgrid(range(0, timesteps), range(-xranges, xranges + 1))

    training_inputs = []
    training_outputs = []
    for x_ in x[:, 0]:
        for t_ in t[0, :]:
            training_inputs.append((float(x_), t_/timesteps))
            training_outputs.append(conduct_draws(proposal, float(x_), t_))

    print(len(training_inputs))
    print(len(training_outputs))
    training_inputs = torch.FloatTensor(np.array(training_inputs)).view(-1, 2)
    training_outputs = torch.FloatTensor(np.array(training_outputs)).view(-1, 2)

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
    logsoftmax = nn.LogSoftmax()
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

def main(args):
    fn_approximator = MLP_factory(input_size=2,
                                  hidden_sizes=args.neural_network,
                                  output_size=2,
                                  hidden_non_linearity=nn.ReLU,
                                  out_non_linearity=None)

    push_toward = [-args.width, args.width]
    X, Y = generate_data(FunnelProposal(push_toward), 60, 50)
    data = TensorDataset(X, Y)
    data = DataLoader(data, batch_size=2048, shuffle=True)
    optimizer = torch.optim.Adam(fn_approximator.parameters(), lr=0.001)
    for i in range(args.epochs):
        losses_for_epoch = []
        for _, (X, Y) in enumerate(data):
            x_mb = X
            y_mb = Y

            y_hat = fn_approximator(x_mb)
            loss = SoftCE(y_hat, y_mb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_for_epoch.append(loss.item())
        if i % 10 == 0:
            print('Update {}, loss {}'.format(i, np.mean(losses_for_epoch)))

    torch.save(CategoricalPolicy(fn_approximator), 'pretrained.pyt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pretraining policies')
    parser.add_argument('-width', '--width', type=int, required=True, help='Width of the Proposal')
    parser.add_argument('-epochs', '--epochs', type=int, required=True, help='Number of epochs to train for')
    parser.add_argument('-nn', '--neural-network', nargs='+', help='neural network specification',
                        default=[32, 32, 32], type=int)

    args = parser.parse_args()
    main(args)