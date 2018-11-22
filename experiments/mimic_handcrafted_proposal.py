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
import mlresearchkit.io as mlio

from pg_methods.networks import MLP_factory
from pg_methods.objectives import PolicyGradientObjective
from pg_methods.policies import CategoricalPolicy
from pg_methods.baselines import FunctionApproximatorBaseline

from rvi_sampling.distributions import proposal_distributions
from rvi_sampling.stochastic_processes import PyTorchWrap
from rvi_sampling.samplers import RVISampler
from rvi_sampling.utils import diagnostics
from rvi_sampling.utils import parsers as rvi_parser
from rvi_sampling.utils import stochastic_processes
from rvi_sampling.utils import analysis

import pretraining_tools

NEURAL_NETWORK = (16, 16)
LEARN_RANGE = 60

def setup_network(args, output_size=2):
    # Setup a network to be trained.
    fn_approximator = MLP_factory(
        input_size=2,
        hidden_sizes=NEURAL_NETWORK,
        output_size=output_size,
        hidden_non_linearity=nn.ReLU,
        out_non_linearity=None)

    return fn_approximator


def run_handcrafted_pretraining(args, save_dir):
    """
    Train the proposal distribution to match a handcrafted distribution.
    :param args:
    :return:
    """
    print('Training a network to mimic hand crafted proposal.')
    # Setup Function Approximator and Optimizer.
    fn_approximator = setup_network(args)
    optimizer = torch.optim.Adam(
        fn_approximator.parameters(),
        lr=args.policy_learning_rate)


    # Setup Proposal to Mimic.
    push_toward = [-args.rw_width, args.rw_width]
    if args.IS_proposal_type == 'soft':
        proposal = proposal_distributions.SimonsSoftProposal(
            push_toward, softness_coeff=args.softness_coefficient)
    elif args.IS_proposal_type == 'funnel':
        proposal = proposal_distributions.FunnelProposal(push_toward)

    print(('Proposal Information:\n'
           'Type: {} Softness Coeff: {}\n'
           'push_toward: {}, time: {}, '
           'learn_range: {}').format(
        args.IS_proposal_type,
        args.softness_coefficient,
        push_toward,
        args.rw_time,
        LEARN_RANGE
    ))

    # Generate the data to train from.
    X, Y = pretraining_tools.generate_data(
        proposal_distributions.FunnelProposal(push_toward),
        args.rw_time,
        LEARN_RANGE)

    data = TensorDataset(X, Y)
    data = DataLoader(data, batch_size=2048, shuffle=True)

    all_losses = []
    for i in range(args.epochs):
        losses_for_epoch = []
        for _, (x_mb, y_mb) in enumerate(data):
            y_hat = fn_approximator(x_mb)
            loss = pretraining_tools.SoftCE(y_hat, y_mb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_for_epoch.append(loss.item())
        if i % 50 == 0:
            print('Epoch: {}/{} Loss {:.5f}'.format(
                i, args.epochs, np.mean(losses_for_epoch)))
        all_losses.append(np.mean(losses_for_epoch))

    file_template = 'pretrained_mimic_{}_{}'.format(
        args.IS_proposal_type, args.softness_coefficient)

    file_to_save = os.path.join(save_dir, file_template + '.pyt')
    mlio.create_folder(save_dir)
    torch.save(CategoricalPolicy(fn_approximator), file_to_save)
    print('Done training. Saved to {}'.format(file_to_save))

    file_to_save = os.path.join(save_dir, file_template + '.summary.txt')
    mlio.put(file_to_save, '\n'.join(map(str, all_losses)))
    sys.exit(0)


def run_multitask_pretraining(args, save_dir):
    """
    Train the proposal distribution to be generally good at all tasks.
    :param args:
    :return:
    """
    args.rw_seed = 5
    rw, analytic = stochastic_processes.create_rw(
        args, biased=False, n_agents=1)
    rw = PyTorchWrap(rw)

    fn_approximator = setup_network(args)
    policy = CategoricalPolicy(fn_approximator)
    policy_optimizer = torch.optim.RMSprop(
        fn_approximator.parameters(),
        lr=args.policy_learning_rate,
        eps=1e-5)

    baseline_fn_approximator = setup_network(args, output_size=1)
    baseline_optimizer = torch.optim.RMSprop(
        baseline_fn_approximator.parameters(),
        lr=args.policy_learning_rate,
        eps=1e-5)
    baseline = FunctionApproximatorBaseline(
        baseline_fn_approximator,
        baseline_optimizer)


    sampler = RVISampler(
        policy,
        policy_optimizer,
        baseline=baseline,
        negative_reward_clip=args.reward_clip,
        objective=PolicyGradientObjective(entropy=args.entropy_coefficient),
        feed_time=True,
        use_gae=(args.gae_value is not None),
        lam=args.gae_value,
        gamma=args.gamma,
        multitask_training=True
    )

    sampler.train(rw, args.epochs, verbose=True)

    file_template = 'pretrained_mimic_{}_{}'.format(
        args.gae_value, args.policy_learning_rate)

    meta_data_folder = os.path.join(save_dir, file_template+'_META')
    mlio.create_folder(save_dir)
    mlio.create_folder(meta_data_folder)

    sampler.set_diagnostic(
        diagnostics.FileSaverHandler([
            diagnostics.EpisodeRewardDiagnostic(10),
            diagnostics.ProportionSuccessDiagnostic(10)],
            meta_data_folder, 'RVISampler', frequency=10)
    )

    file_to_save = os.path.join(save_dir, file_template + '.pyt')

    mlio.argparse_saver(os.path.join(meta_data_folder, 'args.txt'), args)

    torch.save(CategoricalPolicy(fn_approximator), file_to_save)
    print('Done training. Saved to {}'.format(file_to_save))

    analysis.plot_proposal(policy, meta_data_folder)

    sys.exit(0)


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
        '--epochs',
        default=1,
        type=int,
        help='Number of epochs to train for.')

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
        default='handcrafted',
        help=('The type of pretraining to do. '
              'Must be one of {handcrafted, multitask}'))

    parser.add_argument(
        '--dry_run',
        default=False,
        action='store_true'
    )

    rvi_parser.bind_random_walk_arguments(parser, rw_endpoint=False)
    rvi_parser.bind_policy_arguments(
        parser, policy_neural_network=False)
    rvi_parser.bind_rvi_arguments(parser)
    rvi_parser.bind_IS_arguments(parser)
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
        job_display_name='pre_{}'.format(args.pretraining_type))
