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
    parser.add_argument('-baseline', '--baseline_type', default='moving_average')
    parser.add_argument('-notime', '--notime', default=True, action='store_false',
                        help='Do not feed time into the neural network proposal')
    parser.add_argument('-gamma', '--gamma', default=1, type=float, help='discount factor')
    return parser

def experimental_arguments(parser):
    """
    Arguments for setting up the experiments
    :param parser:
    :return:
    """
    parser.add_argument('-s', '--samples', default=1000, type=int, help='number of mc steps')
    parser.add_argument('-samseed', '--sampler_seed', default=0, type=int, help='The seed to use for the samplers')

    parser.add_argument('-n_cpus', '--n_cpus', default=3, type=float,
                        help='CPUs to use when doing the work')
    parser.add_argument('-notb', '--no_tensorboard', action='store_true',
                        help='Disables tensorboard')
    parser.add_argument('-name', '--name', default='', type=str,
                        help='append name')
    parser.add_argument('-outfolder', '--outfolder', default='./', type=str,
                        help='Where to save things')
    return parser

