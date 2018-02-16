import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import multiprocessing
import time
import seaborn as sns
from rvi_sampling.samplers import ISSampler, ABCSampler, MCSampler, RVISampler
from rvi_sampling.StochasticProcess import RandomWalk, PyTorchWrap
from rvi_sampling.distributions.proposal_distributions import SimonsProposal, SimonsSoftProposal
from rvi_sampling.distributions.prior_distributions import DiscreteUniform
from rvi_sampling.plotting import determine_panel_size, visualize_proposal, multi_quiver_plot
from rvi_sampling.distributions.analytic_posterior import TwoStepRandomWalkPosterior
from rvi_sampling.results import ImportanceSamplingResults
from pg_methods.utils.baselines import MovingAverageBaseline
from pg_methods.utils.policies import MultinomialPolicy, RandomPolicy
from pg_methods.utils.networks import MLP_factory
from pg_methods.utils.objectives import PolicyGradientObjective

create_folder = lambda f: [os.makedirs(os.path.join('./', f)) if not os.path.exists(os.path.join('./', f)) else False]
def touch(path):
    with open(path, 'a'):
        os.utime(path, None)
def run_sampler(args):
    sampler, rw, MC_samples = args
    if isinstance(sampler, RVISampler):
        return sampler.solve(PyTorchWrap(rw), MC_SAMPLES)
    else:
        return sampler.solve(rw, MC_SAMPLES)

if __name__=='__main__':

    parser = argparse.ArgumentParser('Comparison of Methods')
    parser.add_argument('-entropy', '--entropy', default=0, type=float, help='entropy coefficient')
    parser.add_argument('-s', '--samples', default=1000, type=int, help='number of mc steps')
    parser.add_argument('-t', '--rw_time', default=50, type=int, help='Length of the random walk')
    parser.add_argument('-seed', '--seed', default=0, type=int, help='The seed to use')
    parser.add_argument('-width', '--rw_width', default=5, type=int,
                        help='width of the discrete uniform in the random walk')
    parser.add_argument('-notime', '--notime', default=True, action='store_false',
                        help='Do not feed time into the neural network proposal')
    parser.add_argument('-baseline_decay', '--baseline_decay', default=0.99, type=float,
                        help='Moving Average baseline decay')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                        help='Learning rate')
    parser.add_argument('-name', '--name', default='', type=str,
                        help='append name')
    parser.add_argument('--only_rvi', default=False, action='store_true',
                        help='does only the RVI experiments')
    parser.add_argument('-n_cpus', '--n_cpus', default=3, type=float,
                        help='CPUs to use when doing the work')

    args = parser.parse_args()

    if args.name != '':
        folder_name = '{}_{}'.format(args.name, time.strftime('%a-%d-%m-%Y__%H-%M-%s'))
    else:
        folder_name = 'results_{}'.format(time.strftime('%a-%d-%m-%Y__%H-%M-%s'))

    sns.set_style('white')
    FEED_TIME = args.notime
    MC_SAMPLES = args.samples
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
                    prior_distribution=DiscreteUniform(DIMENSIONS, -DISC_UNIFORM_WIDTH, 2*DISC_UNIFORM_WIDTH, seed=args.seed+2),
                    seed=args.seed+1)
    rw.reset()

    # create a policy for the RVI sampler
    fn_approximator = MLP_factory(DIMENSIONS+int(FEED_TIME),
                                  hidden_sizes=[32, 32],
                                  output_size=len(POSSIBLE_STEPS),
                                  hidden_non_linearity=nn.ReLU)
    policy = MultinomialPolicy(fn_approximator)
    policy_optimizer = torch.optim.RMSprop(fn_approximator.parameters(),lr=args.learning_rate)


    samplers = [ISSampler(SimonsSoftProposal, seed=args.seed),
                ABCSampler(0,seed=args.seed),
                MCSampler(seed=args.seed),
                RVISampler(policy,
                           policy_optimizer,
                           baseline=MovingAverageBaseline(args.baseline_decay),
                           objective=PolicyGradientObjective(entropy=args.entropy),
                           feed_time=FEED_TIME,
                           seed=args.seed) ]

    if args.only_rvi:
        samplers = [samplers[-1]]

    print('True Starting Position is:{}'.format(rw.x0))
    print('True Ending Position is: {}'.format(rw.xT))

    pool = multiprocessing.Pool(args.n_cpus)
    solver_arguments = [(sampler, rw, MC_SAMPLES) for sampler in samplers]

    sampler_results = pool.map(run_sampler, solver_arguments)

    # for sampler in samplers:
    #     if isinstance(sampler, RVISampler):
    #         sampler_result = sampler.solve(PyTorchWrap(rw), MC_SAMPLES)
    #     else:
    #         sampler_result = sampler.solve(rw, MC_SAMPLES)
    #     print('*'*45)
    #     print('Sampler: {}'.format(sampler._name))
    #     print('Starting Position Estimate: {:3g}, variance: {:3g}'.format(sampler_result.expectation(),
    #                                                                 sampler_result.variance()))
    #     sampler_results.append(sampler_result)


    analytic = TwoStepRandomWalkPosterior(DISC_UNIFORM_WIDTH, 0.5, T)

    panel_size = determine_panel_size(len(sampler_results))
    create_folder(folder_name)
    touch(os.path.join(folder_name, 'start={}'.format(rw.x0)))
    touch(os.path.join(folder_name, 'end={}'.format(rw.xT)))
    fig_dists = plt.figure(figsize=(8, 9))
    fig_traj = plt.figure(figsize=(9,9))
    fig_traj_evol = plt.figure(figsize=(9,9))

    for i, sampler_result in enumerate(sampler_results):
        print(sampler_result.summary())
        ax = fig_dists.add_subplot(panel_size+str(i+1))
        ax = sampler_result.plot_distribution(DISC_UNIFORM_WIDTH, ax, alpha=0.7)
        ax = analytic.plot(rw.xT, ax, label='analytic', color='r')

        title_string = '{}, mean={:3g},\n var={:3g}, %succ={}'.format(sampler_result.sampler_name,
                                                                           sampler_result.expectation(),
                                                                           sampler_result.variance(),
                                                                           len(sampler_result.trajectories())/len(sampler_result.all_trajectories()))
        if isinstance(sampler_result, ImportanceSamplingResults):
            title_string += ' ESS={}'.format(sampler_result.effective_sample_size())

        ax.set_title(title_string)

        ax = fig_traj.add_subplot(panel_size+str(i+1))
        ax = sampler_result.plot_mean_trajectory(ax=ax)
        ax.set_title('Trajectory Distribution\nfor {}'.format(sampler_result.sampler_name))

        ax = fig_traj_evol.add_subplot(panel_size+str(i+1))
        ax = sampler_result.plot_all_trajectory_evolution(ax=ax)
        ax.set_title('Evolution of Trajectories\nfor {}'.format(sampler_result.sampler_name))
        sampler_result.save_results(folder_name)

    fig_dists.suptitle('MC_SAMPLES: {}, Analytic mean: {:3g}, Start {}, End {}'.format(MC_SAMPLES,
                                                                                       analytic.expectation(rw.xT[0]),
                                                                                       rw.x0,
                                                                                       rw.xT),
                       x=0.5,
                       y=1.01)
    fig_dists.tight_layout()
    fig_dists.savefig(os.path.join(folder_name, 'ending_distribution.pdf'))

    fig_traj.tight_layout()
    fig_traj.savefig(os.path.join(folder_name, 'trajectory_distribution.pdf'))

    fig_traj_evol.tight_layout()
    fig_traj_evol.savefig(os.path.join(folder_name, 'trajectory_evolution.pdf'))

    torch.save(policy, os.path.join(folder_name, 'rvi_policy.pyt'))

    t, x, x_arrows, y_arrows_nn = visualize_proposal([policy], 50, 20, neural_network=True)
    f = multi_quiver_plot(t, x, x_arrows,
                          [y_arrows_nn],
                          ['Neural Network Proposal'],
                          figsize=(10, 5))
    f.savefig(os.path.join(folder_name, 'visualized_proposal.pdf'))