from test_tube import argparse_hopt
from test_tube import hpc
from mlresearchkit.computecanada import parsers as compute_canada_parser
from rvi_sampling.utils import parsers as rvi_parser

def main(args):
    pass

def dummy_function(args):
    pass

if __name__ == '__main__':
    # Hyperparameter arguments.
    parser = argparse_hopt.HyperOptArgumentParser(strategy='random_search')
    parser.add_argument(
        '--experiment_name',
        default='test',
    )
    parser.add_argument(
        '--save_dir_tempalte',
        default='/scratch/{experiment_name}/{hyperparameters}',
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

    # Compute Canada specific arguments.
    compute_canada_parser.create_cc_arguments(parser)

    # RVI Specific arguments.
    rvi_parser.random_walk_arguments(parser)
    rvi_parser.rvi_arguments(parser)


    hyperparams = parser.parse_args()

    cluster = hpc.SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=hyperparams.save_dir_template,
        pythom_cmd='python3',
        test_tube_exp_name=hyperparams.experiment_name,
        enable_log_err=True,
        enable_log_out=True
    )

    cluster.notify_job_status(
        email='zafarali.ahmed@mail.mcgill.ca',
        on_done=True,
        on_fail=True)

    cluster.load_modules(['cuda/8.0.44', 'cudnn/7.0'])
    cluster.add_command('source $RVI_ENV')

    cluster.per_experiment_nb_cpus = 2
    cluster.per_experiment_nb_nodes = 1
    cluster.memory_mb_per_node = 16384
    cluster.script_name = ...  # Name of the script here.
    cluster.optimize_parallel_cluster_cpu(
        dummy_function,  # Do nothing here. `script_name` is overwritten.
        nb_trials=24,
        job_name='first_tt_job',
        job_display_name='short_name')
