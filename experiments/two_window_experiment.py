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
from rvi_sampling.distributions.proposal_distributions import SimonsSoftProposal, FunnelProposal
from rvi_sampling import utils
from pg_methods.baselines import MovingAverageBaseline
from pg_methods.policies import MultinomialPolicy
from pg_methods.networks import MLP_factory
from pg_methods.objectives import PolicyGradientObjective

DIMENSIONS = 1
OUTPUT_SIZE = 2

if __name__=='__main__':
    args = utils.parsers.create_parser('1D random walk with two windows', 'random_walk').parse_args()
    utils.common.set_global_seeds(args.sampler_seed)
    folder_name = utils.io.create_folder_name(args.outfolder, args.name)
    utils.io.create_folder(folder_name)

    sns.set_style('white')
    args.windows = [(-args.rw_width, -args.rw_width // 2),
                    (args.rw_width // 2, args.rw_width)]

    rw, analytic = utils.stochastic_processes.create_rw_two_window(args)
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

    push_toward = [-args.rw_width, args.rw_width]
    if args.IS_proposal == 'soft':
        proposal = SimonsSoftProposal(push_toward)
    else:
        proposal = FunnelProposal(push_toward)

    samplers = [ISSampler(proposal),
                ABCSampler('slacked',seed=args.sampler_seed),
                MCSampler(seed=args.sampler_seed),
                RVISampler(policy,
                           policy_optimizer,
                           baseline=baseline,
                           objective=PolicyGradientObjective(entropy=args.entropy),
                           feed_time=args.notime,
                           seed=args.sampler_seed) ]

    if args.only_rvi:
        samplers = [samplers[-1]]

    print('True Starting Position is:{}'.format(rw.x0))
    print('True Ending Position is: {}'.format(rw.xT))


    def kl_function(estimated_distribution):
        return analytic.kl_divergence(estimated_distribution, rw.xT)

    # kl_function = utils.diagnostics.make_kl_function(analytic, rw.xT) Can't work because lambda function
    _ = [sampler.set_diagnostic(utils.diagnostics.create_diagnostic(sampler._name, args, folder_name, kl_function)) for sampler in samplers]


    pool = multiprocessing.Pool(args.n_cpus)
    solver_arguments = [(sampler, utils.stochastic_processes.create_rw_two_window(args)[0], args.samples) for sampler in samplers]

    sampler_results = pool.map(utils.multiprocessing_tools.run_sampler, solver_arguments)
    utils.analysis.analyze_samplers_rw(sampler_results, args, folder_name, rw, policy=policy, analytic=analytic)
