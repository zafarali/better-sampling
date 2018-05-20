"""
This script is used to understand how RVI behaves during and after training
"""
import sys
sys.path.append('../..')
import torch
import matplotlib
matplotlib.use('Agg')
import torch.nn as nn
import numpy as np
import os
import seaborn as sns
from rvi_sampling.samplers import RVISampler
from rvi_sampling.stochastic_processes import PyTorchWrap
from rvi_sampling import utils
from rvi_sampling.samplers import ISSampler, ABCSampler, MCSampler
from rvi_sampling.distributions.proposal_distributions import FunnelProposal

DIMENSIONS = 1
OUTPUT_SIZE = 2
BIASED = False

if __name__=='__main__':
    parser = utils.parsers.create_parser('1D random walk', 'random_walk')
    parser.add_argument('-cycles', '--cycles', type=int, default=15,
                        help='number of train-test cycles.')
    parser.add_argument('-method', '--method', type=str, required=True,
                        help='Baseline method to benchmark')
    args = parser.parse_args()

    utils.common.set_global_seeds(args.sampler_seed)
    sns.set_style('whitegrid')
    folder_name = utils.io.create_folder_name(args.outfolder, args.name+'_'+str(args.sampler_seed)+'_'+str(args.rw_seed)+'_'+str(args.method))

    train_folder_name = os.path.join(folder_name, 'training_results')

    train_folder_to_save_in = os.path.join(train_folder_name, '0')
    utils.io.create_folder(train_folder_to_save_in)

    kl_train_cumulative_track = os.path.join(folder_name, 'kl_training_cumulative.txt')
    kl_train_track = os.path.join(folder_name, 'kl_training.txt')


    prop_train_cumulative_track = os.path.join(folder_name, 'prop_training_cumulative.txt')
    prop_train_track = os.path.join(folder_name, 'prop_training.txt')

    utils.io.create_folder(folder_name)
    utils.io.create_folder(train_folder_name)

    rw, analytic = utils.stochastic_processes.create_rw(args, biased=BIASED, n_agents=args.n_agents)

    utils.io.touch(os.path.join(folder_name, 'start={}'.format(rw.x0)))
    utils.io.touch(os.path.join(folder_name, 'end={}'.format(rw.xT)))

    push_toward = [-args.rw_width, args.rw_width]

    if args.method == 'ISSampler':
        sampler = ISSampler(FunnelProposal(push_toward), seed=args.sampler_seed)
    elif args.method == 'MCSampler':
        sampler = MCSampler(seed=args.sampler_seed)
    elif args.method == 'ABCSampler':
        sampler = ABCSampler('slacked',seed=args.sampler_seed)
    else:
        raise ValueError('Unknown method')


    def kl_function(estimated_distribution):
        return analytic.kl_divergence(estimated_distribution, rw.xT)

    sampler.set_diagnostic(utils.diagnostics.create_diagnostic(sampler._name, args, folder_name, kl_function))

    print('True Starting Position is:{}'.format(rw.x0))
    print('True Ending Position is: {}'.format(rw.xT))
    print('Analytic Starting Position: {}'.format(analytic.expectation(rw.xT[0])))

    test_results = sampler.solve(rw, args.samples, verbose=True)
    train_results = None


    kld = utils.analysis.analyze_samplers_rw([test_results], args, train_folder_to_save_in, rw,
                                       policy=None, analytic=analytic)

    utils.io.put(kl_train_track, '0, '+str(kld[0]))
    utils.io.put(kl_train_cumulative_track, '0, '+str(kld[0]))

    utils.io.put(prop_train_track, '0, ' + str(test_results.prop_success()))
    utils.io.put(prop_train_cumulative_track, '0, ' + str(test_results.prop_success()))


    for i in range(1, args.cycles+1):
        train_results_new = sampler.solve(rw, args.samples)

        # technically doing this saving doesn't take too long so doesn't need to be run
        # in a background thread. This is good because it saves time of having to copy
        # the policy for saving etc.
        if train_results is None:
            train_results = train_results_new
        else:
            # augment the old Results object.
            train_results._all_trajectories.extend(train_results_new.all_trajectories())
            train_results._trajectories.extend(train_results_new.trajectories())
            train_results._posterior_particles = np.hstack([train_results.posterior(),
                                                            train_results_new.posterior()])

            train_results._posterior_weights = np.hstack([train_results.posterior_weights(),
                                                          train_results_new.posterior_weights()])

            train_results.loss_per_episode.extend(train_results_new.loss_per_episode)
            train_results.rewards_per_episode.extend(train_results_new.rewards_per_episode)



        steps_so_far = str(i * args.train_steps)


        train_folder_to_save_in = os.path.join(train_folder_name, str(i))
        utils.io.create_folder(train_folder_to_save_in)
        print('Training Phase:')
        kld = utils.analysis.analyze_samplers_rw([train_results], args, None, rw,
                                           policy=None, analytic=analytic) # don't save these things again

        utils.io.stash(kl_train_cumulative_track, steps_so_far + ', ' + str(kld[0]))
        utils.io.stash(prop_train_cumulative_track, steps_so_far + ', ' + str(train_results.prop_success()))


        kld = utils.analysis.analyze_samplers_rw([train_results_new], args, train_folder_to_save_in, rw,
                                           policy=None, analytic=analytic) # don't save these things again
        utils.io.stash(kl_train_track, steps_so_far + ', ' + str(kld[0]))
        utils.io.stash(prop_train_track, steps_so_far + ', ' + str(train_results_new.prop_success()))
    print('DONE')
