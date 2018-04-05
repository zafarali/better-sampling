"""
Experiments to run on ComputeCanada
to see how our RVI performs over distributions
of different seeds for the walk and samplers.
"""
import sys
import os
sys.path.append('..')
from rvi_sampling import utils
from mlresearchkit.computecanada.slurm import parsers
import argparse


class Experiments(object):
    @staticmethod
    def one_window(args, replicate_id, rw_seed):
        script = '\npython rw_experiment.py -s {samples} -samseed {replicate_id} -n_cpus 1' \
                 '--rw_time $RWTIME --rw_seed {rw_seed} --rw_width $RWWIDTH --outfolder {folder}/{rw_seed} ' \
                 '--n_agents {n_agents} --reward_clip {reward_clip}'
        script = script.format(samples=args.samples,
                               replicate_id=replicate_id,
                               folder=args.out,
                               rw_seed=rw_seed,
                               n_agents=args.n_agents,
                               reward_clip=args.reward_clip)
        return script

    def two_window(args, replicate_id, rw_seed):
        # TODO: two window stuff here.
        script = '\npython two_window_experiment.py -s {samples} -samseed {replicate_id} -n_cpus 1' \
                 '--rw_time $RWTIME --rw_seed {rw_seed} --rw_width $RWWIDTH --outfolder {folder}/{rw_seed} ' \
                 '--n_agents {n_agents} --reward_clip {reward_clip}'
        script = script.format(samples=args.samples,
                               replicate_id=replicate_id,
                               folder=args.out,
                               rw_seed=rw_seed,
                               n_agents=args.n_agents,
                               reward_clip=args.reward_clip)
        return script


def main(args):

    SCRIPT = parsers.create_slurm_header(args)
    SCRIPT += '\nmodule load cuda/8.0.44'
    SCRIPT += '\nmodule load cudnn/7.0'
    SCRIPT += '\nsource $RVI_ENV'
    SCRIPT += '\nRWWIDTH={};'.format(args.rw_width)
    SCRIPT += '\nRWTIME={};'.format(args.rw_time)
    SCRIPT += '\n'
    ## do setup of the experiment here.

    experiment_constructor = getattr(Experiments, args.experiment)
    counter = 0
    for rw_seed in range(args.n_tasks):
        if not args.dryrun: os.mkdir(os.path.join(args.out, str(rw_seed)))
        # each task will be saved in their own folders.
        for replicate_id in range(args.n_replicates):
            SCRIPT += experiment_constructor(args, replicate_id, rw_seed)
            counter += 0
            if counter % args.cc_cpus == 0:
                SCRIPT+='\nwait' # wait until these tasks complete so we can use other cpus after
            else:
                SCRIPT+=' &' # keep putting these on the pid stack.

        SCRIPT += '\nwait'
        # TODO: call some kind of aggregator here for per task aggregation?

    # TODO: call some kind of aggregator here for overall aggregation
    SCRIPT += '\npython INSERT AGGREGATOR HERE!!'
    if args.dryrun:
        print(SCRIPT)
    else:
        utils.io.put(os.path.join(args.out, 'runnable.sh'), SCRIPT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Random Walk Job Creator For Analyzing Performance in Expectation')
    parser.add_argument('-out', '--out', help='Output Folder', default=None)
    parser.add_argument('-exp', '--experiment', help='Name of experiment to run', default='one_window')
    parser = utils.parsers.random_walk_arguments(parser)
    parser.add_argument('-samples', '--samples', default=3000, type=int, help='number of mc steps')
    parser.add_argument('-n_replicates', '--n_replicates', help='Number of replicates to run for each RVI', default=2, type=int)
    parser.add_argument('-n_tasks', '--n_tasks', help='Number of walks/tasks to run', default=2, type=int)
    parser.add_argument('-dryrun', '--dryrun', help='Dry run', default=False, action='store_true')
    parsers.create_slurm_arguments(parser)
    args = parser.parse_args()
    if args.out is None: args.out = './{}'.format(args.experiment)
    if not args.dryrun: os.mkdir(args.out)
    if not args.dryrun: utils.io.argparse_saver(os.path.join(args.out, 'args'), args)
    args.out =  os.path.abspath(args.out)
    args.cc_log = args.out
    main(args)
