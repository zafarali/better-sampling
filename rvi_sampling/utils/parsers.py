import argparse
import logging

from rvi_sampling.utils.stochastic_processes import (
    random_walk_arguments,
    bind_random_walk_arguments  # This unused import is OK.
)


def create_parser(experiment_name, stochastic_process='random_walk'):
    parser = argparse.ArgumentParser('Experiment:' + experiment_name)
    parser = rvi_arguments(parser)
    parser = experimental_arguments(parser)
    if stochastic_process == 'random_walk': parser = random_walk_arguments(parser)
    return parser

def bind_policy_arguments(parser, policy_learning_rate=True, policy_neural_network=True):
    """
    These are arguments needed for setting up networks used as a policy.
    :param parser:
    :return:
    """
    if policy_neural_network:
        parser.add_argument('--policy_neural_network',
                            nargs='+',
                            help='Neural network specification for the policy.',
                            default=(16, 16),
                            type=int, # TODO(zaf): Find a way to pass tuple arguments.
                            # type=lambda s: tuple(map(int, s.split(' ')))
                            )

    parser.add_argument('--pretrained_policy',
                        default=None,
                        type=str,
                        help='Path to a pretrained policy. Default is None.')

    if policy_learning_rate:
        parser.add_argument('--policy_learning_rate',
                            default=0.001,
                            type=float,
                            help='Learning rate for the policy network.')


    return parser



def bind_value_function_arguments(parser,
                                  baseline_learning_rate=True,
                                  baseline_decay_rate=True,
                                  baseline_neural_network=True):
    """
    These are arguments needed for setting up networks used as a baseline.
    :param parser:
    :return:
    """
    parser.add_argument('--baseline_type', default='moving_average')

    if baseline_neural_network:
        parser.add_argument('--baseline_neural_network',
                            nargs='+',
                            help='Neural network specification for the baseline.',
                            default=(16, 16),
                            type=int)

    if baseline_learning_rate:
        parser.add_argument('--baseline_learning_rate',
                            default=0.001,
                            type=float,
                            help='The learning rate for the baseline function.')

    if baseline_decay_rate:
        parser.add_argument('--baseline_decay_rate',
                            default=0.99,
                            type=float,
                            help='The decay rate for the moving average baseline.')

    return parser


def bind_rvi_arguments(parser, n_agents=True, gae_value=True):
    """
    RVI-specific arugments.
    :param parser:
    :return:
    """
    parser.add_argument('--disable_training',
                        default=False,
                        action='store_true',
                        help='Stops training')

    parser.add_argument('--entropy_coefficient',
                        default=0.0,
                        type=float,
                        help='Entropy bonus to encourage policy to be more stochastic.')

    parser.add_argument('--gamma',
                        default=0.99,
                        type=float,
                        help='Discount factor.')

    parser.add_argument('--reward_clip',
                        default=-10,
                        type=float,
                        help=('The value to clip negative infinite rewards to this value. '
                             'If not set, will not clip.'))
    if n_agents:
        parser.add_argument('--n_agents',
                            default=1,
                            type=int,
                            help='Number of agents for each training step.')

    if gae_value:
        parser.add_argument('--gae_value',
                            default=None,
                            type=float,
                            help='Value for gae. If None, no GAE is used.')

    return parser


def bind_sampler_arguments(parser, outfolder=True):
    """
    General sampler arguments.
    :param parser:
    :param outfolder:
    :return:
    """
    parser.add_argument('--samples',
                        default=1000,
                        type=int,
                        help='Total number of monte carlo samples.')


    parser.add_argument('--seed',
                        default=0,
                        type=int,
                        help='The seed to use for the run.')

    parser.add_argument('--use_gpu',
                        default=False,
                        action='store_true',
                        help='Uses GPU if available.')

    parser.add_argument('--no_tensorboard',
                        action='store_true',
                        default=False,
                        help='Disables tensorboard.')

    if outfolder:
        parser.add_argument('--outfolder',
                            default='./',
                            type=str,
                            help='Where to save things.')

    return parser

def bind_IS_arguments(parser, softness_coefficient=True, IS_proposal_type=True):
    """
    Importance sampler arguments.
    :param parser:
    :param softness_coefficient:
    :return:
    """
    if IS_proposal_type:
        parser.add_argument('--IS_proposal_type',
                            default='funnel',
                            type=str,
                            help='The importance sampling distribution to use (funnel, soft)')

    if softness_coefficient:
        parser.add_argument('--softness_coefficient',
                            default=1.0,
                            type=float,
                            help='Sets the softness of the soft proposal in ISSampler.')

    return parser

def rvi_arguments(parser):
    """
    Arguments for the Reinforcement Learning based agent
    :param parser:
    :return:
    """
    logging.warning('rvi_arguments is no longer updated. Use the newer bind_* functions to bind arguments.')

    parser.add_argument('-entropy', '--entropy', default=0, type=float, help='entropy coefficient')
    parser.add_argument('-baseline_decay', '--baseline_decay', default=0.99, type=float,
                        help='Moving Average baseline decay')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                        help='Learning rate')
    parser.add_argument('-baseline_lr', '--baseline_learning_rate', default=0.001, type=float,
                        help='learning rate for baseline function approximator')
    parser.add_argument('--only_rvi', default=False, action='store_true',
                        help='does only the RVI experiments')
    parser.add_argument('--notrain', default=False, action='store_true',
                        help='Stops training')
    parser.add_argument('-baseline', '--baseline_type', default='moving_average')

    # technically notime should be True when we dont want time.
    # here it is messed up a bit and reversed.
    parser.add_argument('-notime', '--notime', default=True, action='store_false',
                        help='Do not feed time into the neural network proposal')
    parser.add_argument('-gamma', '--gamma', default=1, type=float, help='discount factor')
    parser.add_argument('-rewardclip', '--reward_clip', default=-10, type=float, help='The value to clip negative '
                                                                                        'infinite rewards to')
    parser.add_argument('-gae', '--use_gae', default=False, action='store_true',
                        help='Use generalized advantage estimation')
    parser.add_argument('-lam', '--lam', default=1.0, type=float, help='Lambda value for generalized advantages.'
                                                                          'Should be in range [0-1]')
    parser.add_argument('-nagents', '--n_agents', default=1, type=int,
                        help='Number of agents to use')
    parser.add_argument('-plot-posterior', '--plot_posterior', default=False, action='store_true',
                        help='Number of agents to use')
    parser.add_argument('-nn', '--neural-network', nargs='+', help='neural network specification',
                        default=[16, 16], type=int)
    parser.add_argument('-baseline_nn', '--baseline_neural_network', nargs='+', help='baseline neural network specification',
                        default=[16, 16], type=int)
    parser.add_argument('-pretrained', '--pretrained', default=None, type=str, help='path to a pretrained policy.')
    return parser

def experimental_arguments(parser):
    """
    Arguments for setting up the experiments
    :param parser:
    :return:
    """
    parser.add_argument('-s', '--samples', default=1000, type=int, help='number of mc steps')
    parser.add_argument('-samseed', '--sampler_seed', default=0, type=int, help='The seed to use for the samplers')

    parser.add_argument('-n_cpus', '--n_cpus', default=3, type=int,
                        help='CPUs to use when doing the work')
    parser.add_argument('-gpu', '--use_cuda', action='store_true',
                        help='Uses GPU if available')
    parser.add_argument('-notb', '--no_tensorboard', action='store_true',
                        help='Disables tensorboard')
    parser.add_argument('-name', '--name', default='results', type=str,
                        help='append name')
    parser.add_argument('-IS_proposal', '--IS_proposal', default='funnel', type=str,
                        help='the importance sampling distribution to use (funnel, soft)')

    parser.add_argument('-soft_coef', '--softness_coefficient', default=1.0, type=float,
                        help='sets the softness of the soft proposal in ISSampler')

    parser.add_argument('-end_ov', '--override_endpoint', action='store_true', default=False, help='switch to Override endpoint')
    parser.add_argument('-endpoint', '--endpoint', default=1, type=int, help='endpoint if endpoint override is switched on')

    parser.add_argument('-outfolder', '--outfolder', default='./', type=str,
                        help='Where to save things')
    parser.add_argument('-profile', '--profile_performance', default=False, action='store_true',
                        help='Indicates if performance has to be profiled')
    return parser

