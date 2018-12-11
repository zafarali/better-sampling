"""
Test tube script for launching baseline jobs with importance sampling
to compare with RVI.

How to launch on SLURM:
```
python3 tt_IS_baseline.py --cc_account $CC_ACCOUNT --no_tensorboard --samples 5001
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


from rvi_sampling.samplers import ISSampler
from rvi_sampling.distributions import proposal_distributions
from rvi_sampling.utils import analysis as sampler_analysis
from rvi_sampling.utils import parsers as rvi_parser
from rvi_sampling.utils import diagnostics
from rvi_sampling.utils import io as rvi_io
from rvi_sampling.utils import common as common_utils
from rvi_sampling.utils import stochastic_processes

DIMENSIONS = 1
OUTPUT_SIZE = 2
# Make use of backfilling using this slightly messy solutions:
END_POINTS = [0, 12, 24, 36, 48]
TOTAL_END_POINTS = len(END_POINTS)

def run_IS(args, *throwaway):
    # Use Slurm task ID as the environment variable.
    print(args)
    args.rw_seed = args.seed  # Backward compat.
    sampler_seed = int(os.getenv('SLURM_ARRAY_TASK_ID', args.seed))
    print('Sampler Seed: {}'.format(sampler_seed))


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
        run_IS_experiment(args, sampler_seed, end_point)

def run_IS_experiment(args, seed, end_point):
    common_utils.set_global_seeds(seed)

    # this is where the rvi experiment actually runs.
    save_dir = os.path.join(
        args.save_dir_template.format(
            scratch=os.getenv('SCRATCH', './'),
            experiment_name=args.experiment_name,
            softness_coefficient=args.softness_coefficient,
            proposal_type=args.IS_proposal_type,
            end_point=end_point),
        'Seed{}'.format(seed)
    )

    #folder_name = rvi_io.create_folder_name('./', save_dir)

    if args.IS_proposal_type == 'funnel' and args.softness_coefficient > 0.0:
        # Since softness coefficient does not make a difference in
        # this setting, reduce the number of unnecessary runs by just
        # exiting if we are not in the default.
        print('Analysis not run for Funnel with non-zero softness.')
        sys.exit(0)

    rvi_io.create_folder(save_dir)
    rvi_io.create_folder(os.path.join(save_dir, 'ISSampler'))
    rvi_io.argparse_saver(
        os.path.join(save_dir, 'args.txt'), args)

    rw, analytic = stochastic_processes.create_rw(
            args, biased=False, n_agents=args.n_agents)
    rw.xT = np.array([end_point])

    print(rw.xT)
    rvi_io.touch(
        os.path.join(save_dir, 'start={}'.format(rw.x0)))
    rvi_io.touch(
        os.path.join(save_dir, 'end={}'.format(rw.xT)))

    push_toward = [-args.rw_width, args.rw_width]
    if args.IS_proposal_type == 'soft':
        proposal = proposal_distributions.SimonsSoftProposal(
            push_toward, softness_coeff=args.softness_coefficient)
    else:
        proposal = proposal_distributions.FunnelProposal(push_toward)

    sampler = ISSampler(proposal, seed=seed)
    
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
            frequency=50))

    print('True Starting Position is:{}'.format(rw.x0))
    print('True Ending Position is: {}'.format(rw.xT))
    print('Analytic Starting Position: {}'.format(analytic.expectation(rw.xT[0])))

    start_time = time.time()
    sampler_result = sampler.solve(rw, args.samples, verbose=True)
    print('Total time taken: {}'.format(time.time() - start_time))
    
    sampler_result.save_results(save_dir)
    sampler_analysis.analyze_sampler_result(
        sampler_result,
        args.rw_width,
        rw.xT[0],
        analytic=analytic,
        save_dir=os.path.join(save_dir, 'ISSampler'))


if __name__ == '__main__':
    parser = argparse_hopt.HyperOptArgumentParser(
        strategy='grid_search')
    parser.add_argument(
        '--experiment_name',
        default='IS_baseline_experiment',
    )
    parser.add_argument(
        '--save_dir_template',
        default=('{scratch}'
                 '/rvi/IS_baseline_results'
                 '/{experiment_name}'
                 '/end_point{end_point}'
                 '/proposal_type{proposal_type}'
                 '/softness_coefficient{softness_coefficient}')
    )
    parser.opt_list(
        '--softness_coefficient',
        # low=0.1,
        # high=1.5,
        type=float,
        options=[0.0, 0.5, 0.8, 1.0, 1.1, 1.5, 2.0],
        tunable=True,
        # nb_samples=5
    )
    parser.opt_list(
        '--IS_proposal_type',
        options=[
            'funnel',
            'soft'
        ],
        tunable=True
    )
    parser.opt_list(
        '--n_agents',
        options=[10],
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
    rvi_parser.bind_IS_arguments(
        parser, softness_coefficient=False, IS_proposal_type=False)
    cc_parser.create_cc_arguments(parser)


    hyperparams = parser.parse_args()

    if hyperparams.dry_run:
        run_IS(hyperparams)
        sys.exit(0)

    del hyperparams.dry_run
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
        value='0-40',
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
    cluster.job_time = hyperparams.cc_time  # 15 mins.
    cluster.memory_mb_per_node = 16384
    cluster.optimize_parallel_cluster_cpu(
        run_IS,
        nb_trials=350,
        job_name='ImpS hyperparameter search',
        job_display_name='imps_hps_'+ hyperparams.experiment_name)
