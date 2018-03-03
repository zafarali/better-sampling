"""
Utility script to instantiate the different stochastic procceses
"""
import numpy as np
from ..stochastic_processes.random_walk import RandomWalk, DiscreteUniform, RWParameters
from ..distributions.analytic_posterior import TwoStepRandomWalkPosterior, MultiWindowTwoStepRandomwWalkPosterior
from ..distributions.prior_distributions import MultiWindowDiscreteUniform
#### RANDOM WALK

UNBIASED_RW = RWParameters([[-1], [+1]], np.ones(2)/2, 1)
BIASED_RW = RWParameters([[-1], [+1]], [3/4, 1/4], 1)

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

    parser.add_argument('-IS_proposal', '--IS_proposal', default='soft',
                        help='proposal distribution to use in IS (soft, funnel)')
    return parser

def create_rw(args, biased=False):
    """
    Creates an unbiased 1D random walk
    :param args:
    :return:
    """
    if type(biased) == bool:
        if biased:
            POSSIBLE_STEPS, STEP_PROBS, DIMENSIONS = BIASED_RW
        else:
            POSSIBLE_STEPS, STEP_PROBS, DIMENSIONS = UNBIASED_RW
    else:
        POSSIBLE_STEPS, STEP_PROBS, DIMENSIONS = biased

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
    if DIMENSIONS == 1 and not biased:
        analytic = TwoStepRandomWalkPosterior(DISC_UNIFORM_WIDTH, 0.5, T)
    else:
        analytic = None
    return rw, analytic

def create_rw_two_window(args):
    """
    Creates an unbiased 1D random walk
    with two dinwos
    :param args:
    :return:
    """

    T = args.rw_time
    DISC_UNIFORM_WIDTH = args.rw_width
    WINDOWS = args.windows
    # first simulate a random walk

    POSSIBLE_STEPS, STEP_PROBS, DIMENSIONS = UNBIASED_RW

    rw = RandomWalk(DIMENSIONS,
                    STEP_PROBS,
                    POSSIBLE_STEPS,
                    n_agents=1,
                    T=T,
                    prior_distribution=MultiWindowDiscreteUniform(DIMENSIONS, WINDOWS, seed=args.rw_seed+2),
                    seed=args.rw_seed+1)
    rw.reset()
    analytic = MultiWindowTwoStepRandomwWalkPosterior(WINDOWS, 0.5, T)
    return rw, analytic
