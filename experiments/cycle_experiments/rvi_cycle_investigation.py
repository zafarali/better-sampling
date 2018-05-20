"""
This script is used to understand how RVI behaves during and after training
"""
import sys
sys.path.append('..')
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
from pg_methods.baselines import MovingAverageBaseline
from pg_methods.policies import MultinomialPolicy
from pg_methods.networks import MLP_factory
from pg_methods.objectives import PolicyGradientObjective

DIMENSIONS = 1
OUTPUT_SIZE = 2
BIASED = False

if __name__=='__main__':
    parser = utils.parsers.create_parser('1D random walk', 'random_walk')
    parser.add_argument('-cycles', '--cycles', type=int, default=15,
                        help='number of train-test cycles.')
    args = parser.parse_args()

    utils.common.set_global_seeds(args.sampler_seed)
    sns.set_style('whitegrid')
    folder_name = utils.io.create_folder_name(args.outfolder, args.name+'_'+str(args.sampler_seed)+'_'+str(args.rw_seed))

    train_folder_name = os.path.join(folder_name, 'training_results')
    test_folder_name = os.path.join(folder_name, 'testing_results')

    kl_train_cumulative_track = os.path.join(folder_name, 'kl_training_cumulative.txt')
    kl_train_track = os.path.join(folder_name, 'kl_training.txt')
    kl_test_track = os.path.join(folder_name, 'kl_testing.txt')

    prop_train_cumulative_track = os.path.join(folder_name, 'prop_training_cumulative.txt')
    prop_train_track = os.path.join(folder_name, 'prop_training.txt')
    prop_test_track = os.path.join(folder_name, 'prop_testing.txt')

    utils.io.create_folder(folder_name)
    utils.io.create_folder(train_folder_name)
    utils.io.create_folder(test_folder_name)

    rw, analytic = utils.stochastic_processes.create_rw(args, biased=BIASED, n_agents=args.n_agents)

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

    sampler = RVISampler(policy,
                         policy_optimizer,
                         baseline=baseline,
                         negative_reward_clip=args.reward_clip,
                         objective=PolicyGradientObjective(entropy=args.entropy),
                         feed_time=args.notime,
                         train_episodes=-1,
                         seed=args.sampler_seed)

    def kl_function(estimated_distribution):
        return analytic.kl_divergence(estimated_distribution, rw.xT)

    sampler.set_diagnostic(utils.diagnostics.create_diagnostic(sampler._name, args, folder_name, kl_function))

    print('True Starting Position is:{}'.format(rw.x0))
    print('True Ending Position is: {}'.format(rw.xT))
    print('Analytic Starting Position: {}'.format(analytic.expectation(rw.xT[0])))

    test_results = sampler.solve(PyTorchWrap(rw), args.samples, verbose=True, retrain=0, return_train_results=False)
    train_results = None

    test_folder_to_save_in = os.path.join(test_folder_name, '0')
    utils.io.create_folder(test_folder_to_save_in)
    kld = utils.analysis.analyze_samplers_rw([test_results], args, test_folder_to_save_in, rw,
                                       policy=policy, analytic=analytic)

    utils.io.put(kl_train_track, '0, '+str(kld[0]))
    utils.io.put(kl_train_cumulative_track, '0, '+str(kld[0]))
    utils.io.put(kl_test_track, '0, '+str(kld[0]))
    utils.io.put(prop_train_track, '0, ' + str(test_results.prop_success()))
    utils.io.put(prop_train_cumulative_track, '0, ' + str(test_results.prop_success()))
    utils.io.put(prop_test_track, '0, ' + str(test_results.prop_success()))

    for i in range(1, args.cycles+1):
        test_results, train_results_new = sampler.solve(PyTorchWrap(rw), args.samples, verbose=True,
                                                        retrain=args.train_steps, return_train_results=True)


        kld = utils.analysis.analyze_samplers_rw([test_results], args, test_folder_to_save_in, rw,
                                           policy=policy, analytic=analytic)
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


        if i >= args.cycles:
            test_folder_to_save_in = folder_name
        else:
            test_folder_to_save_in = os.path.join(test_folder_name, str(i))
            utils.io.create_folder(test_folder_to_save_in)

        steps_so_far = str(i * args.train_steps)
        print('Testing Phase:')
        kld = utils.analysis.analyze_samplers_rw([test_results], args, test_folder_to_save_in, rw,
                                           policy=policy, analytic=analytic)

        utils.io.stash(kl_test_track, steps_so_far+', ' + str(kld[0]))
        utils.io.stash(prop_test_track, steps_so_far + ', ' + str(test_results.prop_success()))

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
