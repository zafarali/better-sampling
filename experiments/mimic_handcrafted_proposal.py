"""Pretrain proposal distributions.

There are two pre-training schemes:
- Matching the hand-crafted using a supervised learning scheme.
- Reinforcement learning on the distribution of available random walk tasks.

Note that this only works for 1D processes.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from test_tube import argparse_hopt
from test_tube import hpc
from mlresearchkit.computecanada import parsers as cc_parser

from pg_methods.networks import MLP_factory
from pg_methods.policies import CategoricalPolicy

from rvi_sampling.distributions import proposal_distributions
from rvi_sampling.utils import parsers as rvi_parser

DIMENSIONS = 1
OUTPUT_SIZE = 2
NEURAL_NETWORK = (16, 16)

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
        np.array(training_inputs)).view(-1, 2)
    training_outputs = torch.from_numpy(
        np.array(training_outputs)).view(-1, 2)

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

def run_handcrafted_pretraining(args, save_dir):
    """
    Train the proposal distribution to match a handcrafted distribution.
    :param args:
    :return:
    """
    fn_approximator = MLP_factory(
        input_size=2,
        hidden_sizes=args.neural_network,
        output_size=2,
        hidden_non_linearity=nn.ReLU,
        out_non_linearity=None)

    push_toward = [-args.width, args.width]
    X, Y = generate_data(
        proposal_distributions.FunnelProposal(push_toward), 60, 50)
    data = TensorDataset(X, Y)
    data = DataLoader(data, batch_size=2048, shuffle=True)
    optimizer = torch.optim.Adam(fn_approximator.parameters(), lr=0.001)
    for i in range(args.epochs):
        losses_for_epoch = []
        for _, (x_mb, y_mb) in enumerate(data):
            y_hat = fn_approximator(x_mb)
            loss = SoftCE(y_hat, y_mb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_for_epoch.append(loss.item())
        if i % 100 == 0:
            print('Update {}, loss {}'.format(i, np.mean(losses_for_epoch)))

    torch.save(CategoricalPolicy(fn_approximator), 'pretrained.pyt')

def run_multitask_pretraining(args, save_dir):
    pass

def run_pretraining(args, *throwaway):
    # Main entry point.

    save_dir = os.path.join(
        args.save_dir_template.format(
            scratch=os.getenv('SCRATCH', './'),
            experiment_name=args.experiment_name,
            pretraining_type=args.pretraining_type
        )
    )

    if args.pretraining_type == 'handcrafted':
        run_handcrafted_pretraining(args, save_dir)
    elif args.pretraining_type == 'multitask':
        run_multitask_pretraining(args, save_dir)
    else:
        raise ValueError('Unknown pretraining type.')


if __name__ == '__main__':
    parser = argparse_hopt.HyperOptArgumentParser(
        prog='Code for pretraining policies.',
        strategy='grid_search')

    parser.add_argument('--experiment_name', default='pretraining')
    parser.add_argument(
        '--save_dir_template',
        default=('{scratch}'
                 '/rvi/rvi_pretrained'
                 '/{experiment_name}'
                 '/{pretraining_type}')
    )

    parser.add_argument(
        '--pretraining_type',
        type=str,
        help=('The type of pretraining to do. '
              'Must be one of {handcrafted, multitask}'))

    parser.add_argument(
        '--dry_run',
        default=False,
        action='store_true'
    )

    rvi_parser.bind_random_walk_arguments(parser, rw_endpoint=False)
    rvi_parser.bind_policy_arguments(
        parser, policy_learning_rate=True, policy_neural_network=False)
    rvi_parser.bind_rvi_arguments(parser)
    rvi_parser.bind_IS_arguments(
        parser, softness_coefficient=True, IS_proposal_type=True)
    cc_parser.create_cc_arguments(parser)

    args = parser.parse_args()

    if args.dry_run:
        run_pretraining(args)
        sys.exit(0)

    del args.dry_run


    cluster = hpc.SlurmCluster(
        hyperparam_optimizer=args,
        log_path=os.path.join(
            os.getenv('SCRATCH', './'),
            'rvi/tt',
            args.experiment_name),
        python_cmd='python3',
        test_tube_exp_name=args.experiment_name,
        enable_log_err=True,
        enable_log_out=True
    )

    cluster.add_slurm_cmd(
        cmd='account',
        value=args.cc_account,
        comment='Account to run this on.'
    )

    if args.cc_email is not None:
        cluster.notify_job_status(
            email=args.cc_mail,
            on_done=True,
            on_fail=True)

    cluster.load_modules(['cuda/8.0.44', 'cudnn/7.0'])
    cluster.add_command('source $RVI_ENV')

    cluster.per_experiment_nb_cpus = args.cc_cpus  # 1 CPU per job.
    cluster.job_time = args.cc_time
    cluster.memory_mb_per_node = args.cc_mem
    cluster.optimize_parallel_cluster_cpu(
        run_pretraining,
        nb_trials=1,
        job_name='RVI Pretraining',
        job_display_name='pre_{}'.format(pretraining_type))
