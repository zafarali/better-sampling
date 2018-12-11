"""Test if all samplers work in batch mode. Does not check correctness."""
from collections import namedtuple
import os

import torch
import torch.nn as nn

from pg_methods.baselines import MovingAverageBaseline
from pg_methods.policies import CategoricalPolicy
from pg_methods.networks import MLP_factory
from pg_methods.objectives import PolicyGradientObjective

from rvi_sampling import samplers
from rvi_sampling.distributions import proposal_distributions
from rvi_sampling.utils import stochastic_processes
from rvi_sampling.utils import multiprocessing_tools

DIMENSIONS = 1
OUTPUT_SIZE = 2

rwargs = namedtuple('rwargs', 'rw_width rw_time rw_seed')
twrwargs = namedtuple('twrwargs', 'rw_width rw_time rw_seed windows')

def test_issampler_funnel_batch():
    args = rwargs(5, 50, 7)

    rw, analytic = stochastic_processes.create_rw(
            args, biased=False, n_agents=10)

    push_toward = [-5, 5]
    proposal = proposal_distributions.FunnelProposal(push_toward)

    sampler = samplers.ISSampler(proposal, seed=0)

    solver_argument = (
            sampler,
            stochastic_processes.create_rw(
                args, biased=False, n_agents=10)[0],
            100)

    sampler_result = multiprocessing_tools.run_sampler(solver_argument)
    assert sampler_result is not None

    all_trajectories = sampler_result.all_trajectories()
    assert len(all_trajectories) == 1000


def test_issampler_soft_batch():
    args = rwargs(5, 50, 7)

    rw, analytic = stochastic_processes.create_rw(
            args, biased=False, n_agents=10)

    push_toward = [-5, 5]
    proposal = proposal_distributions.SimonsSoftProposal(push_toward)

    sampler = samplers.ISSampler(proposal, seed=0)

    solver_argument = (
            sampler,
            stochastic_processes.create_rw(
                args, biased=False, n_agents=10)[0],
            100)

    sampler_result = multiprocessing_tools.run_sampler(solver_argument)
    assert sampler_result is not None

    all_trajectories = sampler_result.all_trajectories()
    assert len(all_trajectories) == 1000


def test_mcsampler_batch():
    args = rwargs(5, 50, 7)

    rw, analytic = stochastic_processes.create_rw(
            args, biased=False, n_agents=10)

    sampler = samplers.MCSampler()

    solver_argument = (
            sampler,
            stochastic_processes.create_rw(
                args, biased=False, n_agents=10)[0],
            100)

    sampler_result = multiprocessing_tools.run_sampler(solver_argument)
    assert sampler_result is not None

    all_trajectories = sampler_result.all_trajectories()

    assert len(all_trajectories) == 1000


def test_rvisampler_batch():
    args = rwargs(5, 50, 7)

    rw, analytic = stochastic_processes.create_rw(
            args, biased=False, n_agents=10)

    fn_approximator = MLP_factory(DIMENSIONS+1,
                                  hidden_sizes=[16, 16],
                                  output_size=OUTPUT_SIZE,
                                  hidden_non_linearity=nn.ReLU)

    policy = CategoricalPolicy(fn_approximator)
    policy_optimizer = torch.optim.RMSprop(fn_approximator.parameters(),lr=0.001)
    baseline = MovingAverageBaseline(0.99)

    sampler = samplers.RVISampler(
            policy,
            policy_optimizer,
            baseline=baseline,
            negative_reward_clip=-1000,
            objective=PolicyGradientObjective(entropy=0),
            feed_time=True,
            seed=0)

    solver_argument = (
            sampler,
            stochastic_processes.create_rw(
                args, biased=False, n_agents=10)[0],
            100)

    sampler_result = multiprocessing_tools.run_sampler(solver_argument)
    assert sampler_result is not None

    all_trajectories = sampler_result.all_trajectories()
    assert len(all_trajectories) == 1000


def test_issampler_two_window_batch():
    args = twrwargs(5, 50, 7, [(-2, 2)])

    rw, analytic = stochastic_processes.create_rw_two_window(
            args, n_agents=10)

    push_toward = [-5, 5]
    proposal = proposal_distributions.SimonsSoftProposal(push_toward)

    sampler = samplers.ISSampler(proposal, seed=0)

    solver_argument = (
            sampler,
            stochastic_processes.create_rw(
                args, biased=False, n_agents=10)[0],
            100)

    sampler_result = multiprocessing_tools.run_sampler(solver_argument)
    assert sampler_result is not None

    all_trajectories = sampler_result.all_trajectories()
    assert len(all_trajectories) == 1000


def test_mcsampler_two_window_batch():
    args = twrwargs(5, 50, 7, [(-2, 2)])

    rw, analytic = stochastic_processes.create_rw_two_window(
            args, n_agents=10)

    sampler = samplers.MCSampler()

    solver_argument = (
            sampler,
            stochastic_processes.create_rw(
                args, biased=False, n_agents=10)[0],
            100)

    sampler_result = multiprocessing_tools.run_sampler(solver_argument)
    assert sampler_result is not None

    all_trajectories = sampler_result.all_trajectories()

    assert len(all_trajectories) == 1000


def test_rvisampler_two_window_batch():
    args = twrwargs(5, 50, 7, [(-2, 2)])

    rw, analytic = stochastic_processes.create_rw_two_window(
            args,  n_agents=10)

    fn_approximator = MLP_factory(DIMENSIONS+1,
                                  hidden_sizes=[16, 16],
                                  output_size=OUTPUT_SIZE,
                                  hidden_non_linearity=nn.ReLU)

    policy = CategoricalPolicy(fn_approximator)
    policy_optimizer = torch.optim.RMSprop(fn_approximator.parameters(),lr=0.001)
    baseline = MovingAverageBaseline(0.99)

    sampler = samplers.RVISampler(
            policy,
            policy_optimizer,
            baseline=baseline,
            negative_reward_clip=-1000,
            objective=PolicyGradientObjective(entropy=0),
            feed_time=True,
            seed=0)

    solver_argument = (
            sampler,
            stochastic_processes.create_rw(
                args, biased=False, n_agents=10)[0],
            100)

    sampler_result = multiprocessing_tools.run_sampler(solver_argument)
    assert sampler_result is not None

    all_trajectories = sampler_result.all_trajectories()
    assert len(all_trajectories) == 1000


