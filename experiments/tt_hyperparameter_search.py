"""
Test run for doing hyperparameter search.
"""
import os

from test_tube import argparse_hopt
from test_tube import hpc
from mlresearchkit.computecanada import parsers as cc_parser

from rvi_sampling.utils import parsers as rvi_parser
from rvi_sampling.utils import io as rvi_io
from rvi_sampling.utils import common as common_utils
from rvi_sampling.utils import stochastic_processes


def run_rvi(args):
    # Use Slurm task ID as the environment variable.

    sampler_seed = os.getenv('SLURM_ARRAY_TASK_ID', args.sampler_seed)

    # Run RVI for different endpoints.
    if args.rw_time == 10:
        end_points = [2, 4, 6, 8]
    elif args.rw_time == 50:
        end_points = [2, 10, 20, 30, 40, 48]
    else:
        raise ValueError('Unknown rw_time.')

    for end_point in end_points:
        run_rvi_experiment(args, sampler_seed, end_point)

def run_rvi_experiment(args, sampler_seed, end_point):
    common_utils.set_global_seeds(sampler_seed)

    # this is where the rvi experiment actually runs.
    save_dir = os.path.join(
        args.save_dir_template.format(
            scratch=os.getenv('SCRATCH', './'),
            experiment_name='testing_experiment',
            learning_rate=args.learning_rate,
            gave_value=args.gae_value),
        'Seed{}'.format(sampler_seed),
        'EndPoint{}'.format(end_point)
    )

    folder_name = rvi_io.create_folder_name(
        args.outfolder, save_dir)

    rvi_io.create_folder(folder_name)
    rvi_io.argparse_saver(
        os.path.join(folder_name, 'args.txt'), args)

    rw, analytic = stochastic_processes.create_rw(args, biased=False)


if __name__ == '__main__':
    parser = argparse_hopt.HyperOptArgumentParser(
        strategy='random_search')
    parser.add_argument(
        '--experiment_name',
        default='test',
    )
    parser.add_argument(
        '--save_dir_template',
        default=('/{scratch}'
                 '/rvi'
                 '/{experiment_name}'
                 '/lr{learning_rate}'
                 '/gae{gae_value}')
    )
    parser.opt_range(
        '--learning_rate',
        low=0.00001,
        high=0.001,
        type=float,
        log_base=10,
        tunable=True,
    )
    parser.opt_range(
        '--gae_value',
        low=0.95,
        high=0.98,
        type=float,
        tunable=True,
    )

    # RVI Specific arguments.
    rvi_parser.random_walk_arguments(parser)
    cc_parser.create_cc_arguments(parser)


    hyperparams = parser.parse_args()

    cluster = hpc.SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=os.path.join(
            os.getenv('SCRATCH', './'),
            'rvi',
            hyperparams.experiment_name),
        python_cmd='python3',
        test_tube_exp_name=hyperparams.experiment_name,
        enable_log_err=True,
        enable_log_out=True
    )

    # Execute the same experiment 5 times.
    cluster.add_slurm_cmd(
        cmd='array',
        value='0,1,2,3,4',
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

    cluster.per_experiment_nb_cpus = 2
    cluster.memory_mb_per_node = 16384
    cluster.optimize_parallel_cluster_cpu(
        run_rvi_experiment,
        nb_trials=24,
        job_name='first_tt_job',
        job_display_name='short_name')
