import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from rvi_sampling.samplers import ISSampler, ABCSampler, MCSampler, RVISampler
from rvi_sampling.StochasticProcess import RandomWalk, PyTorchWrap
from rvi_sampling.distributions.proposal_distributions import SimonsProposal
from rvi_sampling.distributions.prior_distributions import DiscreteUniform
from rvi_sampling.plotting import determine_panel_size
from rvi_sampling.distributions.analytic_posterior import TwoStepRandomWalkPosterior
from pg_methods.utils.baselines import MovingAverageBaseline
from pg_methods.utils.policies import MultinomialPolicy, RandomPolicy
from pg_methods.utils.networks import MLP_factory

if __name__=='__main__':
    sns.set_style('white')

    MC_SAMPLES = 1000
    POSSIBLE_STEPS = [[-1], [+1]]
    STEP_PROBS = np.ones(2)/2
    DIMENSIONS = 1
    T = 100
    DISC_UNIFORM_WIDTH = 2
    # first simulate a random walk
    rw = RandomWalk(DIMENSIONS,
                    STEP_PROBS,
                    POSSIBLE_STEPS,
                    n_agents=1,
                    T=T,
                    prior_distribution=DiscreteUniform(DIMENSIONS, -DISC_UNIFORM_WIDTH, 2*DISC_UNIFORM_WIDTH))
    rw.reset()

    # create a policy for the RVI sampler
    fn_approximator = MLP_factory(DIMENSIONS,
                                  hidden_sizes=[32, 32],
                                  output_size=len(POSSIBLE_STEPS),
                                  hidden_non_linearity=nn.ReLU)
    policy = MultinomialPolicy(fn_approximator)
    policy_optimizer = torch.optim.RMSprop(fn_approximator.parameters(),lr=0.001)


    samplers = [ISSampler(SimonsProposal),
                ABCSampler(0),
                MCSampler(),
                RVISampler(policy, policy_optimizer, baseline=MovingAverageBaseline(0.99), feed_time=True) ]

    print('True Starting Position is:{}'.format(rw.x0))
    print('True Ending Position is: {}'.format(rw.xT))

    sampler_results = []
    for sampler in samplers:
        if isinstance(sampler, RVISampler):
            sampler_result = sampler.solve(PyTorchWrap(rw), MC_SAMPLES)
        else:
            sampler_result = sampler.solve(rw, MC_SAMPLES)
        print('*'*45)
        print('Sampler: {}'.format(sampler._name))
        weighted = True if isinstance(sampler, (RVISampler, ISSampler)) else False
        print('Starting Position Estimate: {:3g}, variance: {:3g}'.format(sampler_result.expectation(weighted),
                                                                    sampler_result.variance(weighted)))
        sampler_results.append(sampler_result)


    analytic = TwoStepRandomWalkPosterior(DISC_UNIFORM_WIDTH, 0.5, T)

    panel_size = determine_panel_size(len(sampler_results))

    fig_dists = plt.figure(figsize=(8, 9))
    fig_traj = plt.figure(figsize=(9,9))
    fig_traj_evol = plt.figure(figsize=(9,9))

    for i, sampler_result in enumerate(sampler_results):
        ax = fig_dists.add_subplot(panel_size+str(i+1))
        ax = sampler_result.plot_distribution(DISC_UNIFORM_WIDTH, ax, alpha=0.7)
        ax = analytic.plot(rw.xT, ax, label='analytic', color='r')
        weighted = True if isinstance(sampler, (RVISampler, ISSampler)) else False
        ax.set_title('{}, mean={:3g},\n var={:3g}, prop success={}'.format(sampler_result.sampler_name,
                                                                           sampler_result.expectation(weighted),
                                                                           sampler_result.variance(weighted),
                                                                           len(sampler_result.trajectories())/len(sampler_result.all_trajectories())))
        ax = fig_traj.add_subplot(panel_size+str(i+1))
        ax = sampler_result.plot_mean_trajectory(ax=ax)
        ax.set_title('Trajectory Distribution\nfor {}'.format(sampler_result.sampler_name))

        ax = fig_traj_evol.add_subplot(panel_size+str(i+1))
        ax = sampler_result.plot_all_trajectory_evolution(ax=ax)
        ax.set_title('Evolution of Trajectories\nfor {}'.format(sampler_result.sampler_name))


    fig_dists.suptitle('MC_SAMPLES: {}, Analytic Mean: {:3g}'.format(MC_SAMPLES, analytic.expectation(rw.xT[0])))
    fig_dists.tight_layout()
    fig_dists.savefig('ending_distribution.pdf')

    fig_traj.tight_layout()
    fig_traj.savefig('trajectory_distribution.pdf')

    fig_traj_evol.tight_layout()
    fig_traj_evol.savefig('trajectory_evolution.pdf')

torch.save(policy, 'rvi_policy.pyt')