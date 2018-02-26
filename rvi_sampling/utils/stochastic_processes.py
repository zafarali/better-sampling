"""
Utility script to instantiate the different stochastic procceses
"""
import numpy as np
from ..stochastic_processes.random_walk import RandomWalk, DiscreteUniform
from ..distributions.analytic_posterior import TwoStepRandomWalkPosterior
#### RANDOM WALK

def random_walk_arguments(parser):
    """
    Arguments for the stochastic process random walk
    :param parser:
    :return:
    """
    parser.add_argument('-t', '--rw_time', default=50, type=int, help='Length of the random walk')
    parser.add_argument('-rwseed', '--rw_seed', default=0, type=int, help='The seed to use for the random walk')

    parser.add_argument('-width', '--rw_width', default=5, type=int,
                        help='width of the discrete uniform in the random walk')
    return parser

def create_rw(args):
    """
    Creates an unbiased 1D random walk
    :param args:
    :return:
    """
    POSSIBLE_STEPS = [[-1], [+1]]
    STEP_PROBS = np.ones(2)/2
    DIMENSIONS = 1
    T = args.rw_time
    DISC_UNIFORM_WIDTH = args.rw_width
    # first simulate a random walk

    rw = RandomWalk(DIMENSIONS,
                    STEP_PROBS,
                    POSSIBLE_STEPS,
                    n_agents=1,
                    T=T,
                    prior_distribution=DiscreteUniform(DIMENSIONS, -DISC_UNIFORM_WIDTH, 2*DISC_UNIFORM_WIDTH, seed=args.rw_seed+2),
                    seed=args.rw_seed+1)
    rw.reset()
    analytic = TwoStepRandomWalkPosterior(DISC_UNIFORM_WIDTH, 0.5, T)
    return rw, analytic

