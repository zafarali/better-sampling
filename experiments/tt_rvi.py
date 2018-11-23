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
from rvi_sampling.utils import analysis as sampler_analysis
from rvi_sampling.utils import parsers as rvi_parser
from rvi_sampling.utils import diagnostics
from rvi_sampling.utils import io as rvi_io
from rvi_sampling.utils import common as common_utils
from rvi_sampling.utils import stochastic_processes

DIMENSIONS = 1
OUTPUT_SIZE = 2
INCLUDE_TIME = True

# Make use of backfilling using this slightly messy solutions:
END_POINTS = [0, 12, 24, 36, 48]
TOTAL_END_POINTS = len(END_POINTS)
NEURAL_NETWORK = (16, 16)

def get_training_iterations(mc_samples, n_agents):
    return mc_samples // n_agents

def run_rvi(args, *throwaway):
    # Use Slurm task ID as the environment variable.
    print(args)
    args.rw_seed = args.seed  # Backward compat.
    sampler_seed = int(os.getenv('SLURM_ARRAY_TASK_ID', args.seed))


    if args.rw_time == 50:
        end_points = END_POINTS
    else:
        raise ValueError('Unknown rw_time.')
    
    if os.getenv('SLURM_ARRAY_TASK_ID', False):
        # Select end point from [0,6] so that we can
        # use array jobs for backfilling.
        end_point_index = sampler_seed % TOTAL_END_POINTS
        sampler_seed = sampler_seed // TOTAL_END_POINTS
        end_points = [end_points[end_point_index]]

    for end_point in end_points:
        print('#'*30)
        print('Starting next experiment with endpoint: {}'.format(end_point))
        print('#'*30)
        run_rvi_experiment(args, sampler_seed, end_point)
    sys.exit(0)

def run_rvi_experiment(args, sampler_seed, end_point):
    common_utils.set_global_seeds(sampler_seed)

    # this is where the rvi experiment actually runs.
    save_dir = os.path.join(
        args.save_dir_template.format(
            scratch=os.getenv('SCRATCH', './'),
            experiment_name=args.experiment_name,
            learning_rate=args.learning_rate,
            gae_value=args.gae_value,
            n_agents=args.n_agents,
            end_point=end_point),
        'Seed{}'.format(sampler_seed)
    )

    #folder_name = rvi_io.create_folder_name('./', save_dir)
    rvi_io.create_folder(save_dir)
    rvi_io.create_folder(os.path.join(save_dir, 'RVISampler'))
    rvi_io.argparse_saver(
        os.path.join(save_dir, 'args.txt'), args)

    rw, analytic = stochastic_processes.create_rw(args, biased=False, n_agents=args.n_agents)
    rw.xT = np.array([end_point])
    rw = PyTorchWrap(rw)

    print(rw.xT)
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
    kl_function = diagnostics.KL_Function(rw.xT[0], analytic)

    sampler.set_diagnostic(
        diagnostics.create_diagnostic(
            sampler._name,
            args,
            save_dir,
            kl_function,
            frequency=10))

    print('True Starting Position is:{}'.format(rw.x0))
    print('True Ending Position is: {}'.format(rw.xT))
    print('Analytic Starting Position: {}'.format(analytic.expectation(rw.xT[0])))

    start_time = time.time()

    training_iterations = get_training_iterations(args.samples, args.n_agents)
    sampler_result = sampler.train(
        rw, training_iterations, verbose=True)

    print('Total time: {}'.format(time.time() - start_time))
    print('Number of training iterations: {}'.format(training_iterations))

    sampler_result.save_results(save_dir)
    sampler_analysis.analyze_sampler_result(
        sampler_result,
        args.rw_width,
        rw.xT[0],
        analytic=analytic,
        save_dir=os.path.join(save_dir, 'RVISampler'),
        policy=policy)

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
                 '/rvi/rvi_results'
                 '/{experiment_name}'
                 '/end_point{end_point}'
                 '/n_agents{n_agents}'
                 '/lr{learning_rate}'
                 '/gae{gae_value}')
    )
    parser.opt_range(
        '--learning_rate',
        low=0.000005,
        high=0.001,
        type=float,
        log_base=10,
        tunable=True,
        nb_samples=50,
    )
    parser.opt_list(
        '--gae_value',
        options=[0.94, 0.95, 0.96, 0.97],
        type=float,
        tunable=True,
    )
    parser.opt_list(
        '--n_agents',
        options=[1],
        type=int,
        tunable=True,
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
        value='0-20',
        comment='Number of repeats.')

    cluster.add_slurm_cmd(
        cmd='account',
        value=hyperparams.cc_account,
        comment='Account to run this on.'
    )

    cluster.notify_job_status(
        email='zafarali.ahmed@mail.mcgill.ca',
        on_done=True,
        on_fail=True)

    cluster.load_modules(['cuda/8.0.44', 'cudnn/7.0'])
    cluster.add_command('source $RVI_ENV')

    cluster.per_experiment_nb_cpus = 1  # 1 CPU per job.
    cluster.job_time = '0:30:00'  # 30 mins.
    cluster.memory_mb_per_node = 16384
    cluster.optimize_parallel_cluster_cpu(
        run_rvi,
        nb_trials=100,
        job_name='RVI Hyperparameter Search',
        job_display_name='rvi_hps')
