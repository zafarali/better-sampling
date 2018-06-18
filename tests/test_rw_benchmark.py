import torch
import torch.nn as nn
import os
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

def test_issampler():
    """
    This test ensures that issampler works as expected
    :return:
    """

    test_seeds = [ 0, 2, 7 ]

    for rw_seed in test_seeds:
        args = rwargs(5, 50, rw_seed)

        rw, analytic = utils.stochastic_processes.create_rw(args, biased=False)
        utils.common.set_global_seeds(0)
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

        sampler = ISSampler(proposal, seed=0)

        print('True Starting Position is:{}'.format(rw.x0))
        print('True Ending Position is: {}'.format(rw.xT))
        print('Analytic Starting Position: {}'.format(analytic.expectation(rw.xT[0])))

        # Test without gpu support
        solver_argument = (sampler, utils.stochastic_processes.create_rw(args, biased=False, n_agents=1)[0], 1000)

        sampler_result = utils.multiprocessing_tools.run_sampler(solver_argument)

        check_issampler_result(sampler_result, args, analytic, rw, rw_seed)


def test_abcsampler():
    """
    This test ensures that issampler works as expected
    :return:
    """

    test_seeds = [ 0, 2, 7 ]

    for rw_seed in test_seeds:
        args = rwargs(5, 50, rw_seed)

        rw, analytic = utils.stochastic_processes.create_rw(args, biased=False)
        utils.common.set_global_seeds(0)
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

        sampler = ABCSampler('slacked',seed=0)

        print('True Starting Position is:{}'.format(rw.x0))
        print('True Ending Position is: {}'.format(rw.xT))
        print('Analytic Starting Position: {}'.format(analytic.expectation(rw.xT[0])))

        # Test without gpu support
        solver_argument = (sampler, utils.stochastic_processes.create_rw(args, biased=False, n_agents=1)[0], 1000)

        sampler_result = utils.multiprocessing_tools.run_sampler(solver_argument)

        check_abcsampler_result(sampler_result, args, analytic, rw, rw_seed)


def test_mcsampler():
    """
    This test ensures that issampler works as expected
    :return:
    """

    test_seeds = [ 0, 2, 7 ]

    for rw_seed in test_seeds:
        args = rwargs(5, 50, rw_seed)

        rw, analytic = utils.stochastic_processes.create_rw(args, biased=False)
        utils.common.set_global_seeds(0)
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

        sampler = MCSampler(seed=0)

        print('True Starting Position is:{}'.format(rw.x0))
        print('True Ending Position is: {}'.format(rw.xT))
        print('Analytic Starting Position: {}'.format(analytic.expectation(rw.xT[0])))

        # Test without gpu support
        solver_argument = (sampler, utils.stochastic_processes.create_rw(args, biased=False, n_agents=1)[0], 1000)

        sampler_result = utils.multiprocessing_tools.run_sampler(solver_argument)

        check_mcsampler_result(sampler_result, args, analytic, rw, rw_seed)


def test_rvisampler():
    """
    This test ensures that issampler works as expected
    :return:
    """

    test_seeds = [ 0, 2, 7 ]

    for rw_seed in test_seeds:
        args = rwargs(5, 50, rw_seed)

        rw, analytic = utils.stochastic_processes.create_rw(args, biased=False)
        utils.common.set_global_seeds(0)
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

        sampler = RVISampler(policy,
                               policy_optimizer,
                               baseline=baseline,
                               negative_reward_clip=-1000,
                               objective=PolicyGradientObjective(entropy=0),
                               feed_time=True,
                               seed=0)

        print('True Starting Position is:{}'.format(rw.x0))
        print('True Ending Position is: {}'.format(rw.xT))
        print('Analytic Starting Position: {}'.format(analytic.expectation(rw.xT[0])))

        # Test without gpu support
        solver_argument = (sampler, utils.stochastic_processes.create_rw(args, biased=False, n_agents=1)[0], 1000)

        sampler_result = utils.multiprocessing_tools.run_sampler(solver_argument)

        check_rvisampler_result(sampler_result, args, analytic, rw, rw_seed)


def check_issampler_result(sampler_result, args, analytic, rw, rw_seed):
    empirical_distribution = sampler_result.empirical_distribution(histbin_range=args.rw_width)

    kl_divergences = analytic.kl_divergence(empirical_distribution, rw.xT[0])

    assert kl_divergences[0] < 1, 'Basic sanity check for {}'.format(sampler_result.sampler_name)
    assert kl_divergences[0] >= 0, 'non negativity constraint of KL for {}'.format(sampler_result.sampler_name)
    if rw_seed == 0:
        good_performance = 0.0041
    elif rw_seed == 2:
        good_performance = 0.012
    elif rw_seed == 7:
        good_performance = 0.0018
    assert kl_divergences[0] < good_performance, 'Good enough performance for this seed({}) for {}'.format(rw_seed, sampler_result.sampler_name)


def check_abcsampler_result(sampler_result, args, analytic, rw, rw_seed):
    empirical_distribution = sampler_result.empirical_distribution(histbin_range=args.rw_width)

    kl_divergences = analytic.kl_divergence(empirical_distribution, rw.xT[0])

    assert kl_divergences[0] < 1, 'Basic sanity check for {}'.format(sampler_result.sampler_name)
    assert kl_divergences[0] >= 0, 'non negativity constraint of KL for {}'.format(sampler_result.sampler_name)
    if rw_seed == 0:
        good_performance = 0.05
    elif rw_seed == 2:
        good_performance = 0.05
    elif rw_seed == 7:
        good_performance = 0.024
    assert kl_divergences[0] < good_performance, 'Good enough performance for this seed({}) for {}'.format(rw_seed, sampler_result.sampler_name)


def check_mcsampler_result(sampler_result, args, analytic, rw, rw_seed):
    empirical_distribution = sampler_result.empirical_distribution(histbin_range=args.rw_width)

    kl_divergences = analytic.kl_divergence(empirical_distribution, rw.xT[0])

    assert kl_divergences[0] < 1, 'Basic sanity check for {}'.format(sampler_result.sampler_name)
    assert kl_divergences[0] >= 0, 'non negativity constraint of KL for {}'.format(sampler_result.sampler_name)
    if rw_seed == 0:
        good_performance = 0.0023
    elif rw_seed == 2:
        good_performance = 0.035
    elif rw_seed == 7:
        good_performance = 0.00093
    assert kl_divergences[0] < good_performance, 'Good enough performance for this seed({}) for {}'.format(rw_seed, sampler_result.sampler_name)


def check_rvisampler_result(sampler_result, args, analytic, rw, rw_seed):
    empirical_distribution = sampler_result.empirical_distribution(histbin_range=args.rw_width)

    kl_divergences = analytic.kl_divergence(empirical_distribution, rw.xT[0])

    assert kl_divergences[0] < 1, 'Basic sanity check for {}'.format(sampler_result.sampler_name)
    assert kl_divergences[0] >= 0, 'non negativity constraint of KL for {}'.format(sampler_result.sampler_name)
    if rw_seed == 0:
        good_performance = 0.0028
    elif rw_seed == 2:
        good_performance = 0.005
    elif rw_seed == 7:
        good_performance = 0.0067
    assert kl_divergences[0] < good_performance, 'Good enough performance for this seed({}) for {}'.format(rw_seed, sampler_result.sampler_name)