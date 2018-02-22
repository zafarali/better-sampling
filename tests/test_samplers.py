import numpy as np
from rvi_sampling.StochasticProcess import RandomWalk
from rvi_sampling.samplers import MCSampler, ISSampler, RVISampler, ABCSampler
from rvi_sampling.distributions.analytic_posterior import TwoStepRandomWalkPosterior
from rvi_sampling.distributions.prior_distributions import DiscreteUniform
from rvi_sampling.distributions.proposal_distributions import SimonsSoftProposal
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
def check_sampler(sampler, rw):
    results = sampler.solve(rw, 100)
    trajs = results.trajectories()
    for traj in trajs:
        assert traj.shape == rw.true_trajectory.shape
    return results

def check_kl(empirical_dist, rw, is_nan):
    kl1, kl2 = analytic.kl_divergence(empirical_dist, rw.xT, verbose=True)
    if is_nan:
        assert np.isnan(kl1)
        assert np.isnan(kl2)
    else:
        assert not np.isnan(kl1)
        assert not np.isnan(kl2)

def test_MC_Sampler():
    rw.reset()
    results = check_sampler(MCSampler(), rw)
    empirical_dist = results.empirical_distribution(DISC_UNIF_WIDTH)
    check_kl(empirical_dist, rw, False)

def test_ABC_sampler():
    rw.reset()
    result = check_sampler(ABCSampler(0), rw)
    empirical_dist = result.empirical_distribution(DISC_UNIF_WIDTH)
    check_kl(empirical_dist, rw, True)

def test_IS_sampler():
    rw.reset()
    results = check_sampler(ISSampler(SimonsSoftProposal), rw)
    empirical_dist = results.empirical_distribution(DISC_UNIF_WIDTH)
    check_kl(empirical_dist, rw, False)