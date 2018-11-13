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

DIMENSIONS = 1
OUTPUT_SIZE = 2
INCLUDE_TIME = True

def run_rvi(args, *throwaway):
    # Use Slurm task ID as the environment variable.
    print(args)
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
        'EndPoint{}'.format(end_point),
        'Seed{}'.format(sampler_seed)
    )

    #folder_name = rvi_io.create_folder_name('./', save_dir)

    rvi_io.create_folder(save_dir)
    rvi_io.argparse_saver(
        os.path.join(save_dir, 'args.txt'), args)

    rw, analytic = stochastic_processes.create_rw(args, biased=False)
    rw.xT = end_point
    rvi_io.touch(
        os.path.join(save_dir, 'start={}'.format(rw.x0)))
    rvi_io.touch(
        os.path.join(save_dir, 'end={}'.format(rw.xT)))

    #############
    ### SET UP POLICY AND OPTIMIZER.
    #############
    fn_approximator = MLP_factory(
        DIMENSIONS+int(INCLUDE_TIME),
        hidden_sizes=args.neural_network,
        output_size=OUTPUT_SIZE,
        hidden_non_linearity=nn.ReLU)
    policy = MultinomialPolicy(fn_approximator)
    policy_optimizer = torch.optim.RMSprop(
        fn_approximator.parameters(),
        lr=args.learning_rate,
        epsilon=1e-5)

    #############
    ### SET UP VALUE FUNCTION AND OPTIMIZER.
    #############
    baseline_fn_approximator = MLP_factory(
        DIMENSIONS+int(INCLUDE_TIME),
        hidden_sizes=args.baseline_neural_network,
        output_size=1,
        hidden_non_linearity=nn.ReLU)
    baseline_optimizer = torch.optim.RMSprop(
        baseline_fn_approximator.parameters(),
        lr=args.baseline_learning_rate,
        epsilon=1e-5)
    baseline = FunctionApproximatorBaseline(
        baseline_fn_approximator,
        baseline_optimizer)

    sampler = RVISampler(
        policy,
        policy_optimizer,
        baseline=baseline,
        negative_reward_clip=args.reward_clip,
        objective=PolicyGradientObjective(entropy=args.entropy),
        feed_time=INCLUDE_TIME,
        seed=sampler_seed,
        use_gae=args.use_gae,
        lam=args.lam,
        gamma=args.gamma)

if __name__ == '__main__':
    parser = argparse_hopt.HyperOptArgumentParser(
        strategy='random_search')
    parser.add_argument(
        '--experiment_name',
        default='test',
    )
    parser.add_argument('--sampler_seed', default=0, type=int)
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
        run_rvi,
        nb_trials=24,
        job_name='first_tt_job',
        job_display_name='short_name')
