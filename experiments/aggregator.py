"""
Utility to aggregate different experimental runs
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

def main(args):
    search_path = os.path.join(args.folder, args.name+'*__*')
    folders = glob(search_path)
    print('Looking in {}'.format(search_path))
    print('Found {} experimental runs'.format(len(folders)))

    empirical_distributions = defaultdict(lambda: {'prob_estimates':[], 'support': []})
    dfs = []
    for folder in folders:
        try:
            KL, args_path = glob(os.path.join(folder, 'KL'))[0], glob(os.path.join(folder, 'args'))[0]
        except Exception as e:
            print('folder: {} has no information saved'.format(folder))
            continue

        # first gather KL-divergences
        data = dict()
        with open(KL, 'r') as f:
            for line in f.readlines():
                a, b = line.strip().split(',')
                data[a+'_KL'] = float(b)

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
            _, sampler_name = res.split('trajectory_results_')
            with open(res, 'r') as f:
                results = json.load(f)

            probs, support = empirical_distribution(results['posterior_particles'],
                                                    results['posterior_weights'],
                                                    histbin_range=max(results['posterior_particles']),
                                                    return_numpy=True)

            empirical_distributions[sampler_name]['prob_estimates'].append(probs)
            empirical_distributions[sampler_name]['support'].append(support)



    df = pd.concat(dfs, ignore_index=True)
    if not args.dryrun: df.to_csv(os.path.join(args.folder, args.save))
    if not args.dryrun: df.mean().to_csv(os.path.join(args.folder, 'MEANS.csv'))
    if not args.dryrun and args.histogram:
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
    parser.add_argument('-s', '--save', help='File to save to', required=False, default='aggregated.csv')
    parser.add_argument('-dry', '--dryrun', help='Dry run', required=False, action='store_true', default=False)
    parser.add_argument('-histogram', '--histogram', help='Creates histograms from the data', required=False, action='store_true', default=False)


    args = parser.parse_args()
    main(args)