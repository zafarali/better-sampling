import torch
import torch.nn as nn
import os
import multiprocessing
import seaborn as sns
from rvi_sampling.samplers import ISSampler, ABCSampler, MCSampler, RVISampler
from rvi_sampling.distributions.proposal_distributions import SimonsSoftProposal, FunnelProposal
from rvi_sampling import utils
from pg_methods.baselines import MovingAverageBaseline
from pg_methods.policies import CategoricalPolicy
from pg_methods.networks import MLP_factory
from pg_methods.objectives import PolicyGradientObjective

DIMENSIONS = 1
OUTPUT_SIZE = 2

from collections import namedtuple

rwargs = namedtuple('rwargs', 'rw_width rw_time rw_seed')

def test_algorithms():
    """
    This test basically ensures progress does not get over time.
    :return:
    """
    args = rwargs(5, 50, 0)

    rw, analytic = utils.stochastic_processes.create_rw(args, biased=False)

    # create a policy for the RVI sampler
    fn_approximator = MLP_factory(DIMENSIONS+1,
                                  hidden_sizes=[16, 16],
                                  output_size=OUTPUT_SIZE,
                                  hidden_non_linearity=nn.ReLU)

    policy = CategoricalPolicy(fn_approximator)
    policy_optimizer = torch.optim.RMSprop(fn_approximator.parameters(),lr=0.001)
    baseline = MovingAverageBaseline(0.99)

    push_toward = [-5, 5]
    proposal = FunnelProposal(push_toward)

    samplers = [ISSampler(proposal, seed=0),
                ABCSampler('slacked',seed=0),
                MCSampler(seed=0),
                RVISampler(policy,
                           policy_optimizer,
                           baseline=baseline,
                           negative_reward_clip=-1000,
                           objective=PolicyGradientObjective(entropy=0),
                           feed_time=True,
                           train_episodes=1000,
                           seed=0) ]



    print('True Starting Position is:{}'.format(rw.x0))
    print('True Ending Position is: {}'.format(rw.xT))
    print('Analytic Starting Position: {}'.format(analytic.expectation(rw.xT[0])))

    pool = multiprocessing.Pool(2)
    solver_arguments = [(sampler, utils.stochastic_processes.create_rw(args, biased=False, n_agents=1)[0], 1000) for sampler in samplers]

    sampler_results = pool.map(utils.multiprocessing_tools.run_sampler, solver_arguments)

    for sampler_result in sampler_results:
        check_sampler_result(sampler_result, args, analytic, rw)


def check_sampler_result(sampler_result, args, analytic, rw):
    empirical_distribution = sampler_result.empirical_distribution(histbin_range=args.rw_width)

    kl_divergences = analytic.kl_divergence(empirical_distribution, rw.xT[0])

    assert kl_divergences[0] < 1, 'Basic sanity check for {}'.format(sampler_result.sampler_name)
    assert kl_divergences[0] >= 0, 'non negativity constraint of KL for {}'.format(sampler_result.sampler_name)
    if sampler_result.sampler_name == 'ABCSampler':
        good_performance = 0.03
    else:
        good_performance = 0.01
    assert kl_divergences[0] < good_performance, 'Good enough performance for this seed for {}'.format(sampler_result.sampler_name)


