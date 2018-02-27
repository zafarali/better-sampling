"""
Experiments to run on compute canada for the RandomWalk
"""
import sys
import os
sys.path.append('..')
from rvi_sampling import utils
from mlresearchkit.computecanada.slurm import parsers
import argparse


class Experiments(object):
    @staticmethod
    def entropy_two_window(args, replicate_id):
        script = ''
        for entropy in [0, 0.1, 0.5, 1, 1.5, 2, 2.5, 3]:
            script += '\npython two_window_experiment.py -entropy {entropy} -n_cpus 3 -s {samples}' \
                      ' -samseed {replicate_id} --rw_time $RWTIME --rw_seed $RWSEED --rw_width $RWWIDTH'\
                      ' --outfolder {folder} -name entropy{name}'
            if args.only_rvi: script += ' --only_rvi'
            script = script.format(entropy=entropy, samples=args.samples, replicate_id=replicate_id,
                                   folder=args.out, name=entropy)

        return script

    @staticmethod
    def gamma_two_window(args, replicate_id):
        script = ''
        for gamma in [0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]:
            script += '\npython two_window_experiment.py -gamma {gamma} -n_cpus 3 -s {samples}' \
                      ' -samseed {replicate_id} --rw_time $RWTIME --rw_seed $RWSEED --rw_width $RWWIDTH'\
                      ' --outfolder {folder} -name gamma{name}'
            if args.only_rvi: script += ' --only_rvi'
            script = script.format(gamma=gamma, samples=args.samples, replicate_id=replicate_id,
                                   folder=args.out, name=gamma)

        return script

    @staticmethod
    def test(args, replicate_id):
        script = '\npython rw_experiment.py -s {samples} -samseed {replicate_id} ' \
                 '--rw_time $RWTIME --rw_seed $RWSEED --rw_width $RWWIDTH --outfolder {folder}'
        if args.only_rvi: script += ' --only_rvi'
        script = script.format(samples=args.samples, replicate_id=replicate_id, folder=args.out)
        return script

    @staticmethod
    def one_window(args, replicate_id):
        script = '\npython rw_experiment.py'

def main(args):
    SCRIPT = parsers.create_slurm_header(args)
    SCRIPT += '\nmodule load python/3.5.2'
    SCRIPT += '\nmodule load cuda/8.0.44'
    SCRIPT += '\nmodule load cudnn/7.0'
    SCRIPT += '\nsource $RVI_ENV'
    SCRIPT += '\nRWWIDTH={};'.format(args.rw_width)
    SCRIPT += '\nRWTIME={};'.format(args.rw_time)
    SCRIPT += '\nRWSEED={};'.format(args.rw_seed)
    SCRIPT += '\n'
    ## do setup of the experiment here.

    for i in range(args.replicates):
        experiment_constructor = getattr(Experiments, args.experiment)
        SCRIPT += experiment_constructor(args, i)

    #TODO: add aggregator here?
    if args.dryrun:
        print(SCRIPT)
    else:
        utils.io.put(os.path.join(args.out, 'runnable.sh'), SCRIPT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Random Walk Job Creator')
    parser.add_argument('-out', '--out', help='Output Folder', default='./randomwalk')
    parser.add_argument('-seed', '--seed', help='Seeds for the samplers. Each run will have seed+reps',
                        default=4, type=int)
    parser.add_argument('-exp', '--experiment', help='Name of experiment to run', default='test')
    parser = utils.parsers.random_walk_arguments(parser)
    parser.add_argument('-samples', '--samples', default=1000, type=int, help='number of mc steps')
    parser.add_argument('-reps', '--replicates', help='Number of replicates to run', default=2, type=int)
    parser.add_argument('-only_rvi', '--only_rvi', help='Only run the RVI model', default=False, action='store_true')
    parser.add_argument('-dryrun', '--dryrun', help='Dry run', default=False, action='store_true')
    parsers.create_slurm_arguments(parser)
    args = parser.parse_args()
    if not args.dryrun: os.mkdir(args.out)
    if not args.dryrun: utils.io.argparse_saver(os.path.join(args.out, 'args'), args)
    args.out =  os.path.abspath(args.out)
    args.cc_log = args.out
    main(args)
