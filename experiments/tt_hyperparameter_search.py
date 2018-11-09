"""
Test run for doing hyperparameter search.
"""
import os

from test_tube import argparse_hopt
from test_tube import hpc
from mlresearchkit.computecanada import parsers as cc_parser

from rvi_sampling.utils import parsers as rvi_parser
from rvi_sampling.utils import io as rvi_io

def run_rvi(args):
    # Use Slurm task ID as the environment variable.

    sampler_seed = os.getenv('SLURM_ARRAY_TASK_ID', args.sampler_seed)
    save_dir = os.path.join(
        args.save_dir_template.format(
            os.getenv('SCRATCH', './'),
            'testing_experiment',
            args.learning_rate,
            args.gae_value),
        'Seed{}'.format(sampler_seed)
    )

    folder_name = rvi_io.create_folder_name(
        args.outfolder, save_dir)

    rvi_io.create_folder(folder_name)
    rvi_io.argparse_saver(
        os.path.join(folder_name, 'args.txt'), args)

    print(args)

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
    )
    parser.opt_range(
        '--gae_value',
        low=0.95,
        high=0.98,
        type=float,
    )

    # RVI Specific arguments.
    rvi_parser.random_walk_arguments(parser)
    cc_parser.create_cc_arguments(parsers)


    hyperparams = parser.parse_args()

    cluster = hpc.SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=os.path.join(
            os.getenv('SCRATCH', './'),
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
    )_

    cluster.notify_job_status(
        email='zafarali.ahmed@mail.mcgill.ca',
        on_done=True,
        on_fail=True)

    cluster.load_modules(['cuda/8.0.44', 'cudnn/7.0'])
    cluster.add_command('source $RVI_ENV')

    cluster.per_experiment_nb_cpus = 2
    cluster.per_experiment_nb_nodes = 1
    cluster.memory_mb_per_node = 16384
    cluster.optimize_parallel_cluster_cpu(
        run_rvi,
        nb_trials=24,
        job_name='first_tt_job',
        job_display_name='short_name')
