"""
A more general utility to summarize multiple runs.
NOTE: this is not backward compatible with previous version of aggregator.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
import os
import argparse
import json
import pandas as pd
import numpy as np
from glob import glob
from rvi_sampling.utils.stats_utils import empirical_distribution, average_estimated_distributions
from collections import defaultdict
import seaborn as sns
sns.set_style('whitegrid')
sns.set_color_codes('colorblind')
sns.set_context('paper', font_scale=1.5)
def main(args):
    search_path = os.path.join(args.folder, args.name+'*__*')
    folders = glob(search_path)
    print('Looking in {}'.format(search_path))
    print('Found {} experimental runs'.format(len(folders)))

    empirical_distributions = defaultdict(lambda: {'prob_estimates':[], 'support': []})
    dfs = [] # this will hold all the arguments

    time_series = defaultdict(lambda: {'KL':[], 'time': []})
    time_series_dfs = []
    kl_values = defaultdict(lambda: [])
    for repeat_number, folder in enumerate(folders):
        try:
            KL, args_path = glob(os.path.join(folder, 'KL'))[0], glob(os.path.join(folder, 'args'))[0]

        except Exception as e: # TODO: catch a more specific exception here.
            print('folder: {} has no information saved. Error was {}'.format(folder, e))
            continue

        # first gather KL-divergences
        data = dict()
        with open(KL, 'r') as f:
            for line in f.readlines():
                a, b = line.strip().split(',')
                data[a+'_KL'] = float(b)
                kl_values[a].append(float(b))

        # now gather the arguments
        with open(args_path, 'r') as f:
            for line in f.readlines():
                a, b = line.strip().split(',')[:2]
                data[a] = b
        # print(data)
        dfs.append(pd.DataFrame(data, index=[0]))

        # now gather the distributions
        trajectory_results = glob(os.path.join(folder, 'trajectory_results_*'))
        for res in trajectory_results:
            if args.no_distribution:
                continue
            _, sampler_name = res.split('trajectory_results_')
            with open(res, 'r') as f:
                results = json.load(f)

            probs, support = empirical_distribution(results['posterior_particles'],
                                                    results['posterior_weights'],
                                                    histbin_range=max(results['posterior_particles']),
                                                    return_numpy=True)

            empirical_distributions[sampler_name]['prob_estimates'].append(probs)
            empirical_distributions[sampler_name]['support'].append(support)

        time_series_files = glob(os.path.join(folder, '*_KLpq.txt'))
        for time_series_file in time_series_files:
            sampler_name = time_series_file.split('_KLpq.txt')[0].split('/')[-1]
            opened_df = pd.read_csv(time_series_file,names=['KL', 'time'])
            opened_df['repeat_id'] = repeat_number
            opened_df['sampler_name'] = sampler_name
            time_series[sampler_name]['KL'].append(opened_df['KL'].values.tolist())
            time_series[sampler_name]['time'].append(opened_df['time'].values.tolist())
            time_series_dfs.append(opened_df)

    if not args.dryrun:
        with open(os.path.join(args.folder, 'KL'), 'w') as f:
            for sampler_name in kl_values.keys():
                f.write('{},{}\n'.format(sampler_name, np.mean(kl_values[sampler_name])))

    # note that this does not require having time counts to be the same
    time_series_dfs = pd.concat(time_series_dfs, ignore_index=True)
    if not args.dryrun:
        f = plt.figure()
        ax = f.add_subplot(1, 1,1)
        sns.tsplot(time_series_dfs, time='time', unit='repeat_id', condition='sampler_name', value='KL', ax=ax)
        ax.semilogy()
        ax.legend(fontsize='x-small')
        f.savefig(os.path.join(args.folder, 'summarized_KL_timeseries.pdf'))
        time_series_dfs.to_csv(os.path.join(args.folder, 'KL_timeseries.csv'))

    # do some kind of summarization for the KL series here.
    # since we have verified that the time counts are always the same
    # what we could do is just average each idx..

    # note that for time series sumamrizer to work, all the times have to be the same.
    for time_series_key in time_series.keys():
        # a round about way to check equality that the time series are all unique
        assert np.all(np.equal(
                            np.unique(time_series[time_series_key]['time']),
                                      time_series[time_series_key]['time'][0]))
        if not args.dryrun:
            # TODO Deal with the Nans in a smarter way?
            with open(os.path.join(args.folder, '{}_KLpq.txt'.format(time_series_key)), 'w') as f:
                for kl_, time_ in zip(np.mean(np.array(time_series[time_series_key]['KL']), axis=0).tolist(),\
                    np.mean(np.array(time_series[time_series_key]['time']), axis=0).tolist()):
                    f.write('{},{}\n'.format(kl_, time_))


    df = pd.concat(dfs, ignore_index=True)
    if not args.dryrun: df.to_csv(os.path.join(args.folder, args.save))
    if not args.dryrun: df.mean().to_csv(os.path.join(args.folder, 'args'))
    if not args.dryrun and args.histogram and not args.no_distribution:
        for sampler_name in empirical_distributions.keys():
            support = np.sort(np.unique(empirical_distributions[sampler_name]['support']))
            stacked_dist = np.stack(empirical_distributions[sampler_name]['prob_estimates'])
            df = pd.DataFrame(stacked_dist,
                              columns=support)

            df.to_csv(os.path.join(args.folder, args.save.replace('.csv', '_histogram_{}.json'.format(sampler_name))))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax = sns.barplot(data=df, color="salmon", saturation=0.5, ax=ax)
            ax.set_title('Aggregated histogram for\n{} ({} replicates)'.format(sampler_name, stacked_dist.shape[0]))
            ax.set_xlabel('Support')
            ax.set_ylabel('Prob')
            fig.savefig(os.path.join(args.folder, args.save.replace('.csv', '_histogram_{}.pdf'.format(sampler_name))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Aggregate experimental runs')
    parser.add_argument('-f', '--folder', help='Folder with the experimental runs', required=True)
    parser.add_argument('-n', '--name', help='Name appenditure for the experimental runs', required=False, default='')
    # parser.add_argument('-c', '--count', help='Count of the experimental run', required=False, default=0, type=int)
    parser.add_argument('-s', '--save', help='File to save to', required=False, default='aggregated.csv')
    parser.add_argument('-dry', '--dryrun', help='Dry run', required=False, action='store_true', default=False)
    parser.add_argument('-no-distribution', '--no-distribution', help='Will skip making a summary distribution',
                        required=False, action='store_true', default=False)
    parser.add_argument('-histogram', '--histogram', help='Creates histograms from the data', required=False, action='store_true', default=False)


    args = parser.parse_args()
    main(args)