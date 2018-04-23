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
BIASED = False

if __name__=='__main__':
    args = utils.parsers.create_parser('1D random walk', 'random_walk').parse_args()
    utils.common.set_global_seeds(args.sampler_seed)
    sns.set_style('white')
    folder_name = utils.io.create_folder_name(args.outfolder, args.name+'_'+str(args.sampler_seed)+'_'+str(args.rw_seed))
    utils.io.create_folder(folder_name)

    rw, analytic = utils.stochastic_processes.create_rw(args, biased=BIASED)
    utils.io.touch(os.path.join(folder_name, 'start={}'.format(rw.x0)))
    utils.io.touch(os.path.join(folder_name, 'end={}'.format(rw.xT)))

    if args.pretrained is not None:
        policy = torch.load(args.pretrained)
        policy_optimizer = torch.optim.RMSprop(policy.fn_approximator.parameters(),lr=args.learning_rate)

    else:
        # create a policy for the RVI sampler
        fn_approximator = MLP_factory(DIMENSIONS+int(args.notime),
                                      hidden_sizes=args.neural_network,
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

    samplers = [ISSampler(proposal, seed=args.sampler_seed),
                ABCSampler('slacked',seed=args.sampler_seed),
                MCSampler(seed=args.sampler_seed),
                RVISampler(policy,
                           policy_optimizer,
                           baseline=baseline,
                           negative_reward_clip=args.reward_clip,
                           objective=PolicyGradientObjective(entropy=args.entropy),
                           feed_time=args.notime,
                           seed=args.sampler_seed) ]
    samplers[-1].train_mode(False)
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
    solver_arguments = [(sampler,
                         utils.stochastic_processes.create_rw(args,
                                                              biased=BIASED,
                                                              n_agents=args.n_agents if sampler._name == 'RVISampler' else 1)[0],
                         # args.samples * args.n_agents if sampler._name != 'RVISampler' else args.samples) for sampler in samplers]
                         args.samples) for sampler in samplers]

    sampler_results = pool.map(utils.multiprocessing_tools.run_sampler, solver_arguments)

    utils.analysis.analyze_samplers_rw(sampler_results, args, folder_name, rw, policy=policy, analytic=analytic)

