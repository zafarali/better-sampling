import argparse
from .stochastic_processes import random_walk_arguments

def create_parser(experiment_name, stochastic_process='random_walk'):
    parser = argparse.ArgumentParser('Experiment:' + experiment_name)
    parser = rvi_arguments(parser)
    parser = experimental_arguments(parser)
    if stochastic_process == 'random_walk': parser = random_walk_arguments(parser)
    return parser

def rvi_arguments(parser):
    """
    Arguments for the Reinforcement Learning based agent
    :param parser:
    :return:
    """
    parser.add_argument('-entropy', '--entropy', default=0, type=float, help='entropy coefficient')
    parser.add_argument('-baseline_decay', '--baseline_decay', default=0.99, type=float,
                        help='Moving Average baseline decay')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                        help='Learning rate')
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
    parser.add_argument('-nagents', '--n_agents', default=1, type=int,
                        help='Number of agents to use')
    parser.add_argument('-trainsteps', '--train_steps', default=1000, type=int,
                        help='Number of steps to train the RVI sampler before sampling.')
    parser.add_argument('-plot-posterior', '--plot_posterior', default=False, action='store_true',
                        help='Number of agents to use')
    parser.add_argument('-nn', '--neural-network', nargs='+', help='neural network specification',
                        default=[32, 32], type=int)
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
    parser.add_argument('-notb', '--no_tensorboard', action='store_true',
                        help='Disables tensorboard')
    parser.add_argument('-name', '--name', default='results', type=str,
                        help='append name')
    parser.add_argument('-IS_proposal', '--IS_proposal', default='funnel', type=str,
                        help='the importance sampling distribution to use (funnel, soft)')
    parser.add_argument('-outfolder', '--outfolder', default='./', type=str,
                        help='Where to save things')
    return parser

