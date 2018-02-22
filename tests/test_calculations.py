import numpy as np
from rvi_sampling.StochasticProcess import RandomWalk
from rvi_sampling.samplers import ISSampler
from rvi_sampling.distributions.analytic_posterior import TwoStepRandomWalkPosterior
from rvi_sampling.distributions.prior_distributions import DiscreteUniform
from rvi_sampling.distributions.proposal_distributions import RandomProposal
POSSIBLE_STEPS = [[-1], [+1]]
STEP_PROBS = [1/2, 1/2]
DIMENSIONS = 1
T = 4
DISC_UNIF_WIDTH = 4

rw = RandomWalk(DIMENSIONS,
                STEP_PROBS,
                POSSIBLE_STEPS,
                T=T,
                prior_distribution=DiscreteUniform(DIMENSIONS, -DISC_UNIF_WIDTH, 2*DISC_UNIF_WIDTH))
analytic = TwoStepRandomWalkPosterior(DISC_UNIF_WIDTH,0.5, T)

def test_IS_ratio_math():
    """
    With a random proposal distribution the probability of a
    forward transition is the same as a backward transition
    therefore, most of the terms in this will cancel out from
    the likelihood ratio and result in a log ratio of 0
    which corresponds to a weight of 1
    (i.e. there needs to be no correction)
    :return:
    """
    iss_sampler = ISSampler(RandomProposal)
    results = iss_sampler.solve(rw, 100)
    posterior_weights = results.posterior_weights()

    assert np.allclose(posterior_weights, 1)

