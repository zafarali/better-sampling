"""
RVI Launcher script.

How to launch on SLURM:
```
python3 tt_rvi.py --cc_account $CC_ACCOUNT --no_tensorboard --samples 5001
```

"""
import sys
import os
import time

import numpy as np
import torch

from test_tube import argparse_hopt
from test_tube import hpc
from mlresearchkit.computecanada import parsers as cc_parser

from pg_methods.baselines import FunctionApproximatorBaseline
from pg_methods.policies import CategoricalPolicy
from pg_methods.networks import MLP_factory
from pg_methods.objectives import PolicyGradientObjective

from rvi_sampling.samplers import RVISampler
from rvi_sampling.stochastic_processes import PyTorchWrap
from rvi_sampling.stochastic_processes.random_walk import RWParameters
from rvi_sampling.utils import analysis as sampler_analysis
from rvi_sampling.utils import parsers as rvi_parser
from rvi_sampling.utils import diagnostics
from rvi_sampling.utils import io as rvi_io
from rvi_sampling.utils import common as common_utils
from rvi_sampling.utils import stochastic_processes

DIMENSIONS = 2
OUTPUT_SIZE = 4
INCLUDE_TIME = True

# Make use of backfilling using this slightly messy solutions:
#END_POINTS = [0, 12, 24, 36, 48]
#TOTAL_END_POINTS = len(END_POINTS)
NEURAL_NETWORK = (32, 32, 32)

# Get 2D random walk.
def get_stochastic_process(args):
    # TODO(zaf): Adhoc, move this somewhere else.
    #rwargs = namedtuple('rwargs', 'rw_width rw_time rw_seed')
    #args = rwargs(5, 50, 0)

    ACTION_PROB = np.ones(4) / 4.0
    #ACTIONS = [[-1, 0], [0, -1], [0, +1], [+1, 0]]
    ACTIONS = [[-1, +1], [-1, -1], [+1, +1], [+1, -1]]
    bias = RWParameters(ACTIONS, ACTION_PROB, DIMENSIONS)

    rw, analytic = stochastic_processes.create_rw(
        args=args,
        biased=bias,
        n_agents=1)

    return (rw, analytic)

def get_training_iterations(mc_samples, n_agents):
    return mc_samples // n_agents

def run_rvi(args, *throwaway):
    # Use Slurm task ID as the environment variable.
    print(args)
    args.rw_seed = args.seed  # Backward compat.
    sampler_seed = int(os.getenv('SLURM_ARRAY_TASK_ID', args.seed))

    run_rvi_experiment(args, sampler_seed)
    sys.exit(0)

def run_rvi_experiment(args, sampler_seed):
    common_utils.set_global_seeds(sampler_seed)

    # this is where the rvi experiment actually runs.
    save_dir = os.path.join(
        args.save_dir_template.format(
            scratch=os.getenv('SCRATCH', './'),
            experiment_name=args.experiment_name,
            learning_rate=args.learning_rate,
            gae_value=args.gae_value,
            n_agents=args.n_agents,
            #end_point=end_point),
            ),
        'Seed{}'.format(sampler_seed)
    )

    #folder_name = rvi_io.create_folder_name('./', save_dir)
    rvi_io.create_folder(save_dir)
    rvi_io.create_folder(os.path.join(save_dir, 'RVISampler'))
    rvi_io.argparse_saver(
        os.path.join(save_dir, 'args.txt'), args)

    rw, analytic = get_stochastic_process(args)
    rw.simulate()
    rw = PyTorchWrap(rw)
    print('end point set to: {}'.format(rw.xT))

    rvi_io.touch(
        os.path.join(save_dir, 'start={}'.format(rw.x0)))
    rvi_io.touch(
        os.path.join(save_dir, 'end={}'.format(rw.xT)))

    #############
    ### SET UP POLICY AND OPTIMIZER.
    #############
    if args.pretrained_policy is not None:
        print('Loaded pretrained policy from: {}'.format(
            args.pretrained_policy))
        policy = torch.load(args.pretrained_policy)
        policy_optimizer = torch.optim.RMSprop(
            policy.fn_approximator.parameters(),
            lr=args.learning_rate,
            eps=1e-5)
    else:
        fn_approximator = MLP_factory(
            DIMENSIONS+int(INCLUDE_TIME),
            hidden_sizes=NEURAL_NETWORK,
            output_size=OUTPUT_SIZE,
            hidden_non_linearity=torch.nn.ReLU)
        policy = CategoricalPolicy(fn_approximator)
        policy_optimizer = torch.optim.RMSprop(
            fn_approximator.parameters(),
            lr=args.learning_rate,
            eps=1e-5)

    #############
    ### SET UP VALUE FUNCTION AND OPTIMIZER.
    #############
    baseline_fn_approximator = MLP_factory(
        DIMENSIONS+int(INCLUDE_TIME),
        hidden_sizes=NEURAL_NETWORK,
        output_size=1,
        hidden_non_linearity=torch.nn.ReLU)
    baseline_optimizer = torch.optim.RMSprop(
        baseline_fn_approximator.parameters(),
        lr=args.learning_rate,
        eps=1e-5)
    baseline = FunctionApproximatorBaseline(
        baseline_fn_approximator,
        baseline_optimizer)

    sampler = RVISampler(
        policy,
        policy_optimizer,
        baseline=baseline,
        negative_reward_clip=args.reward_clip,
        objective=PolicyGradientObjective(entropy=args.entropy_coefficient),
        feed_time=INCLUDE_TIME,
        seed=sampler_seed,
        use_gae=(args.gae_value is not None),
        lam=args.gae_value,
        gamma=args.gamma)

    if args.disable_training:
        print('Training has been disabled.')
        sampler.train_mode(False)
        rw.train_mode(False)

    #####################
    ### Statistics
    #####################
    kl_function = diagnostics.KL_Function(rw.xT, analytic)

    sampler.set_diagnostic(
        diagnostics.create_diagnostic(
            sampler._name,
            args,
            save_dir,
            kl_function,
            frequency=5))

    print('True Starting Position is:{}'.format(rw.x0))
    print('True Ending Position is: {}'.format(rw.xT))
    #print('Analytic Starting Position: {}'.format(
    #    analytic.expectation(rw.xT[0])))

    start_time = time.time()

    training_iterations = get_training_iterations(args.samples, args.n_agents)
    print('Number of training iterations: {}'.format(training_iterations))

    sampler_result = sampler.train(
        rw, training_iterations, verbose=True)

    print('Total time: {}'.format(time.time() - start_time))

    sampler_result.save_results(save_dir)
    #sampler_analysis.analyze_sampler_result(
    #    sampler_result,
    #    args.rw_width,
    #    rw.xT[0],
    #    analytic=analytic,
    #    save_dir=os.path.join(save_dir, 'RVISampler'),
    #    policy=policy)

if __name__ == '__main__':
    parser = argparse_hopt.HyperOptArgumentParser(
        strategy='grid_search')
    parser.add_argument(
        '--experiment_name',
        default='rvi_experiment',
    )
    parser.add_argument(
        '--save_dir_template',
        default=('{scratch}'
                 '/rvi/2d_results_indep'
                 '/{experiment_name}'
                 #'/end_point{end_point}'
                 '/n_agents{n_agents}'
                 '/lr{learning_rate}'
                 '/gae{gae_value}')
    )
    parser.add_argument(
        '--n_windows',
        default=1,
        type=int
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=1,
        help='Number of hyperparameter trials'
    )
    parser.opt_list(
        '--learning_rate',
        options=[0.00001],
        type=float,
        tunable=True,
    )
    parser.opt_list(
        '--gae_value',
        options=[0.9],
        type=float,
        tunable=True,
    )
    parser.add_argument(
        '--n_agents',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--dry_run',
        default=False,
        action='store_true'
    )

    # Setup general argparse arguments.
    rvi_parser.bind_random_walk_arguments(parser, rw_endpoint=False)
    rvi_parser.bind_sampler_arguments(parser, outfolder=False)
    rvi_parser.bind_policy_arguments(
        parser, policy_learning_rate=False, policy_neural_network=False)
    rvi_parser.bind_value_function_arguments(
        parser, baseline_learning_rate=False, baseline_neural_network=False)
    rvi_parser.bind_rvi_arguments(parser, n_agents=False, gae_value=False)
    cc_parser.create_cc_arguments(parser)


    hyperparams = parser.parse_args()

    array_def = '0-10'

    if hyperparams.dry_run:
        run_rvi(hyperparams)
        sys.exit(0)

    del hyperparams.dry_run

    # # TODO(zaf): Figure out how to fix passing tuples.
    # # To subprocesses.
    # # Remove these commands since they cause failures.
    # del hyperparams.policy_neural_network
    # del hyperparams.baseline_neural_network

    cluster = hpc.SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=os.path.join(
            os.getenv('SCRATCH', './'),
            'rvi/tt',
            hyperparams.experiment_name),
        python_cmd='python3',
        test_tube_exp_name=hyperparams.experiment_name,
        enable_log_err=True,
        enable_log_out=True
    )

    # Execute the same experiment 5 times.
    cluster.add_slurm_cmd(
        cmd='array',
        value=array_def,
        comment='Number of repeats.')

    cluster.add_slurm_cmd(
        cmd='account',
        value=hyperparams.cc_account,
        comment='Account to run this on.'
    )

    if hyperparams.cc_mail is not None:
        cluster.notify_job_status(
            email=hyperparams.cc_mail,
            on_done=True,
            on_fail=True)

    cluster.load_modules(['cuda/8.0.44', 'cudnn/7.0'])
    cluster.add_command('source $RVI_ENV')

    cluster.per_experiment_nb_cpus = 1  # 1 CPU per job.
    cluster.job_time = hyperparams.cc_time  # 30 mins.
    cluster.memory_mb_per_node = 16384
    cluster.optimize_parallel_cluster_cpu(
        run_rvi,
        nb_trials=hyperparams.n_trials,
        job_name='RVI 2D',
        job_display_name='2d_rvi{}_{}'.format(
            hyperparams.n_windows,
            hyperparams.experiment_name))
