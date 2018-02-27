import sys
sys.path.append('..')
import torch
import matplotlib
matplotlib.use('Agg')
import torch.nn as nn
import os
import multiprocessing
import seaborn as sns
from rvi_sampling.samplers import ISSampler, ABCSampler, MCSampler, RVISampler
from rvi_sampling.distributions.proposal_distributions import SimonsSoftProposal
from rvi_sampling import utils
from pg_methods.utils.baselines import MovingAverageBaseline
from pg_methods.utils.policies import MultinomialPolicy
from pg_methods.utils.networks import MLP_factory
from pg_methods.utils.objectives import PolicyGradientObjective

DIMENSIONS = 1
OUTPUT_SIZE = 2
BIASED = False

if __name__=='__main__':
    args = utils.parsers.create_parser('1D random walk', 'random_walk').parse_args()
    utils.common.set_global_seeds(args.sampler_seed)
    sns.set_style('white')
    folder_name = utils.io.create_folder_name(args.name)
    utils.io.create_folder(folder_name)

    rw, analytic = utils.stochastic_processes.create_rw(args, biased=BIASED)
    utils.io.touch(os.path.join(folder_name, 'start={}'.format(rw.x0)))
    utils.io.touch(os.path.join(folder_name, 'end={}'.format(rw.xT)))

    # create a policy for the RVI sampler
    fn_approximator = MLP_factory(DIMENSIONS+int(args.notime),
                                  hidden_sizes=[16, 16],
                                  output_size=OUTPUT_SIZE,
                                  hidden_non_linearity=nn.ReLU)

    policy = MultinomialPolicy(fn_approximator)
    policy_optimizer = torch.optim.RMSprop(fn_approximator.parameters(),lr=args.learning_rate)
    baseline = MovingAverageBaseline(args.baseline_decay)

    samplers = [ISSampler(SimonsSoftProposal, seed=args.sampler_seed),
                ABCSampler(0,seed=args.sampler_seed),
                MCSampler(seed=args.sampler_seed),
                RVISampler(policy,
                           policy_optimizer,
                           baseline=baseline,
                           objective=PolicyGradientObjective(entropy=args.entropy),
                           feed_time=args.notime,
                           seed=args.sampler_seed) ]

    if args.only_rvi:
        samplers = [samplers[-1]]

    def kl_function(estimated_distribution):
        return analytic.kl_divergence(estimated_distribution, rw.xT)

    # kl_function = utils.diagnostics.make_kl_function(analytic, rw.xT) Can't work because lambda function
    _ = [sampler.set_diagnostic(utils.diagnostics.create_diagnostic(sampler._name, args, folder_name, kl_function)) for sampler in samplers]

    print('True Starting Position is:{}'.format(rw.x0))
    print('True Ending Position is: {}'.format(rw.xT))
    print('Analytic Starting Position: {}'.format(analytic.expectation(rw.xT[0])))

    pool = multiprocessing.Pool(args.n_cpus)
    solver_arguments = [(sampler, utils.stochastic_processes.create_rw(args, biased=BIASED)[0], args.samples) for sampler in samplers]

    sampler_results = pool.map(utils.multiprocessing_tools.run_sampler, solver_arguments)

    utils.analysis.analyze_samplers_rw(sampler_results, args, folder_name, rw, policy=policy, analytic=analytic)

