import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import random
import multiprocessing
import time
import pickle
import seaborn as sns
from rvi_sampling.samplers import ISSampler, ABCSampler, MCSampler, RVISampler
from rvi_sampling.StochasticProcess import RandomWalk, PyTorchWrap
from rvi_sampling.distributions.proposal_distributions import SimonsSoftProposal
from rvi_sampling.distributions.prior_distributions import DiscreteUniform
from rvi_sampling.plotting import determine_panel_size, visualize_proposal, multi_quiver_plot
from rvi_sampling.distributions.analytic_posterior import TwoStepRandomWalkPosterior
from pg_methods.utils.baselines import MovingAverageBaseline, NeuralNetworkBaseline
from pg_methods.utils.policies import MultinomialPolicy
from pg_methods.utils.networks import MLP_factory
from pg_methods.utils.objectives import PolicyGradientObjective
from rvi_sampling.utils.diagnostics import (KLDivergenceDiagnostic,
                                            TensorBoardHandler,
                                            ProportionSuccessDiagnostic,
                                            DiagnosticHandler)

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
    parser.add_argument('-rwseed', '--rw_seed', default=0, type=int, help='The seed to use for the random walk')
    parser.add_argument('-samseed', '--sampler_seed', default=0, type=int, help='The seed to use for the samplers')

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
    parser.add_argument('-notb', '--no_tensorboard', action='store_true',
                        help='Disables tensorboard')
    parser.add_argument('-baseline', '--baseline_type', default='moving_average')

    args = parser.parse_args()

    torch.manual_seed(args.sampler_seed)
    np.random.seed(args.sampler_seed)
    random.seed(args.sampler_seed)
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

    def create_rw():
        rw = RandomWalk(DIMENSIONS,
                        STEP_PROBS,
                        POSSIBLE_STEPS,
                        n_agents=1,
                        T=T,
                        prior_distribution=DiscreteUniform(DIMENSIONS, -DISC_UNIFORM_WIDTH, 2*DISC_UNIFORM_WIDTH, seed=args.rw_seed+2),
                        seed=args.rw_seed+1)
        rw.reset()
        return rw

    rw = create_rw()
    analytic = TwoStepRandomWalkPosterior(DISC_UNIFORM_WIDTH, 0.5, T)
    create_folder(folder_name)
    touch(os.path.join(folder_name, 'start={}'.format(rw.x0)))
    touch(os.path.join(folder_name, 'end={}'.format(rw.xT)))

    def kl_function(estimated_distribution):
        return analytic.kl_divergence(estimated_distribution, rw.xT)

    def create_diagnostic(sampler_name):
        diagnostics = [KLDivergenceDiagnostic(kl_function, DISC_UNIFORM_WIDTH, 5), ProportionSuccessDiagnostic(5)]
        if args.no_tensorboard:
            diagnostic_handler = DiagnosticHandler(diagnostics)
        else:
            print('Tensorboard Logging at: {}'.format(os.path.join(folder_name, sampler_name)))
            diagnostic_handler = TensorBoardHandler(diagnostics,log_dir=os.path.join(folder_name, sampler_name))

        return diagnostic_handler

    print(create_diagnostic('bla'))

    # create a policy for the RVI sampler
    fn_approximator = MLP_factory(DIMENSIONS+int(FEED_TIME),
                                  hidden_sizes=[16, 16],
                                  output_size=len(POSSIBLE_STEPS),
                                  hidden_non_linearity=nn.ReLU)
    policy = MultinomialPolicy(fn_approximator)
    policy_optimizer = torch.optim.RMSprop(fn_approximator.parameters(),lr=args.learning_rate)

    if args.baseline_type == 'moving_average':
        baseline = MovingAverageBaseline(args.baseline_decay)
    elif args.baseline_type == 'neural_network':

        value_fn_approximator = MLP_factory(DIMENSIONS+int(FEED_TIME),
                                            hidden_sizes=[16, 16],
                                            output_size=1,
                                            hidden_non_linearity=nn.ReLU)

        value_fn_optimizer = torch.optim.RMSprop(fn_approximator.parameters(), lr=args.learning_rate)

        baseline = NeuralNetworkBaseline(value_fn_approximator,
                                         value_fn_optimizer)

    else:
        raise KeyError('Unknown baseline type')


    print('length of trajectory: {}'.format(rw.true_trajectory.shape))
    samplers = [ISSampler(SimonsSoftProposal, seed=args.sampler_seed),
                ABCSampler(0,seed=args.sampler_seed),
                MCSampler(seed=args.sampler_seed),
                RVISampler(policy,
                           policy_optimizer,
                           baseline=baseline,
                           objective=PolicyGradientObjective(entropy=args.entropy),
                           feed_time=FEED_TIME,
                           seed=args.sampler_seed) ]

    if args.only_rvi:
        samplers = [samplers[-1]]

    _ = [sampler.set_diagnostic(create_diagnostic(sampler._name)) for sampler in samplers]

    print('True Starting Position is:{}'.format(rw.x0))
    print('True Ending Position is: {}'.format(rw.xT))

    pool = multiprocessing.Pool(args.n_cpus)
    solver_arguments = [(sampler, create_rw(), MC_SAMPLES) for sampler in samplers]

    sampler_results = pool.map(run_sampler, solver_arguments)


    print('Analytic Starting Position: {}'.format(analytic.expectation(rw.xT[0])))
    panel_size = determine_panel_size(len(sampler_results))
    create_folder(folder_name)
    touch(os.path.join(folder_name, 'start={}'.format(rw.x0)))
    touch(os.path.join(folder_name, 'end={}'.format(rw.xT)))

    fig_dists = plt.figure(figsize=(8, 9))
    fig_traj = plt.figure(figsize=(9,9))
    fig_traj_evol = plt.figure(figsize=(9,9))
    fig_weight_hists = plt.figure(figsize=(9,4))

    hist_colors = zip(['r', 'g', 'b'], [1, 2, 3])


    for i, sampler_result in enumerate(sampler_results):
        ax = fig_dists.add_subplot(panel_size+str(i+1))
        ax = sampler_result.plot_distribution(DISC_UNIFORM_WIDTH, ax, alpha=0.7)
        ax = analytic.plot(rw.xT, ax, label='analytic', color='r')

        empirical_distribution = sampler_result.empirical_distribution(DISC_UNIFORM_WIDTH)
        kl_divergence = analytic.kl_divergence(empirical_distribution, rw.xT[0])
        ax.set_title(sampler_result.summary_title() + '\nKL(true|est)={:3g}, KL(est|true)={:3g}'.format(*kl_divergence))
        print(sampler_result.summary('KL(true|est)={:3g}, KL(obs|est)={:3g}'.format(*kl_divergence)))
        ax = fig_traj.add_subplot(panel_size+str(i+1))
        ax = sampler_result.plot_mean_trajectory(ax=ax)
        ax.set_title('Trajectory Distribution\nfor {}'.format(sampler_result.sampler_name))

        ax = fig_traj_evol.add_subplot(panel_size+str(i+1))
        ax = sampler_result.plot_all_trajectory_evolution(ax=ax)
        ax.set_title('Evolution of Trajectories\nfor {}'.format(sampler_result.sampler_name))
        sampler_result.save_results(folder_name)


        if sampler_result._importance_sampled:
            c, j = next(hist_colors)
            ax = fig_weight_hists.add_subplot('12'+str(j))
            sampler_result.plot_posterior_weight_histogram(ax, color=c, label='{}'.format(sampler_result.sampler_name))
            ax.legend()

    fig_weight_hists.tight_layout()
    fig_weight_hists.savefig(os.path.join(folder_name, 'weight_distribution.pdf'))

    fig_dists.suptitle('MC_SAMPLES: {}, Analytic mean: {:3g}, Start {}, End {}'.format(MC_SAMPLES,
                                                                                       analytic.expectation(rw.xT[0]),
                                                                                       rw.x0,
                                                                                       rw.xT))
    fig_dists.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig_dists.savefig(os.path.join(folder_name, 'ending_distribution.pdf'))

    fig_traj.tight_layout()
    fig_traj.savefig(os.path.join(folder_name, 'trajectory_distribution.pdf'))

    fig_traj_evol.tight_layout()
    fig_traj_evol.savefig(os.path.join(folder_name, 'trajectory_evolution.pdf'))

    fig_weight_hists.tight_layout()
    torch.save(policy, os.path.join(folder_name, 'rvi_policy.pyt'))

    t, x, x_arrows, y_arrows_nn = visualize_proposal([policy], 50, 20, neural_network=True)
    f = multi_quiver_plot(t, x, x_arrows,
                          [y_arrows_nn],
                          ['Neural Network Proposal'],
                          figsize=(10, 5))
    f.savefig(os.path.join(folder_name, 'visualized_proposal.pdf'))

    pickle.dump(args, open(os.path.join(folder_name, 'args.pkl'), 'wb'))
