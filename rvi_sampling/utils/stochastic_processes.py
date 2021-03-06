"""
Utility script to instantiate the different stochastic procceses
"""
import logging
import numpy as np
from ..stochastic_processes.random_walk import RandomWalk, DiscreteUniform, RWParameters
from ..stochastic_processes.epidemiology import SIR, SIRParameters
from ..distributions.analytic_posterior import (
        TwoStepRandomWalkPosterior,
        MultiWindowTwoStepRandomwWalkPosterior,
        MultiDimensionalRandomWalkPosterior)
from ..distributions.prior_distributions import MultiWindowDiscreteUniform
from ..distributions.arbitriary_priors import ArbitriaryPrior

#### RANDOM WALK
UNBIASED_RW = RWParameters([[-1], [+1]], np.ones(2)/2, 1)
BIASED_RW = RWParameters([[-1], [+1]], [3/4, 1/4], 1)

def random_walk_arguments(parser):
    """
    Arguments for the stochastic process random walk
    :param parser:
    :return:
    """
    logging.warning('Use bind_random_walk_arguments going forward.')
    parser.add_argument('-t', '--rw_time', default=50, type=int, help='Length of the random walk')
    parser.add_argument('-rwseed', '--rw_seed', default=0, type=int, help='The seed to use for the random walk')

    parser.add_argument('-width', '--rw_width', default=5, type=int,
                        help='width of the discrete uniform in the random walk')
    return parser

def bind_random_walk_arguments(parser, rw_width=True, rw_time=True, rw_endpoint=True, rw_windows=True):
    if rw_time:
        parser.add_argument('--rw_time',
                            default=50,
                            type=int,
                            help='Length of the random walk')
    if rw_endpoint:
        parser.add_argument('--rw_endpoint',
                            default=0,
                            type=int,
                            help='The location of the end point.')

    if rw_width:
        parser.add_argument('--rw_width',
                            default=5,
                            type=int,
                            help='width of the discrete uniform in the random walk')
    if rw_windows:
        parser.add_argument('--rw_windows',
                            default=None,
                            type=list,
                            help='window opening definition')
    return parser


def create_rw(args, biased=False, n_agents=1):
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
                    n_agents=n_agents,
                    T=T,
                    prior_distribution=DiscreteUniform(DIMENSIONS, -DISC_UNIFORM_WIDTH, 2*DISC_UNIFORM_WIDTH, seed=args.rw_seed+2),
                    seed=args.rw_seed+1)
    rw.reset()
    if DIMENSIONS == 1 and not biased:
        analytic = TwoStepRandomWalkPosterior(DISC_UNIFORM_WIDTH, 0.5, T)
    elif DIMENSIONS > 1:
        analytic = MultiDimensionalRandomWalkPosterior(
                DISC_UNIFORM_WIDTH, 0.5, T, dimensions=DIMENSIONS)
    else:
        analytic = None
    return rw, analytic

def create_rw_two_window(args, n_agents=1):
    """
    Creates an unbiased 1D random walk
    with two dinwos
    :param args:
    :return:
    """

    T = args.rw_time
    DISC_UNIFORM_WIDTH = args.rw_width
    WINDOWS = args.rw_windows
    # first simulate a random walk

    POSSIBLE_STEPS, STEP_PROBS, DIMENSIONS = UNBIASED_RW

    rw = RandomWalk(DIMENSIONS,
                    STEP_PROBS,
                    POSSIBLE_STEPS,
                    n_agents=n_agents,
                    T=T,
                    prior_distribution=MultiWindowDiscreteUniform(DIMENSIONS, WINDOWS, seed=args.rw_seed+2),
                    seed=args.rw_seed+1)
    rw.reset()
    analytic = MultiWindowTwoStepRandomwWalkPosterior(WINDOWS, 0.5, T)
    return rw, analytic

### SIR
DEFAULT_SIR = SIRParameters(population_size=100, infection_rate=1, recovery_rate=0.3, T=1500, delta_t=0.01)

def create_SIR(args, sir_params=DEFAULT_SIR, prior=ArbitriaryPrior(np.array([[98, 2]])), n_agents=1):
    """

    :param args:
    :return:
    """

    sir = SIR(sir_params.population_size,
              sir_params.infection_rate,
              sir_params.recovery_rate,
              prior,
              sir_params.T,
              sir_params.delta_t,
              seed=args.sir_seed,
              n_agents=n_agents)


    sir.reset()

    return sir, None

