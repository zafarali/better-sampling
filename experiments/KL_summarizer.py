"""
A utility to plot a histogram of KL divergences as a nice plot
this is useful for showing performance of a method on average.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import os
import pandas as pd
import argparse
import numpy as np
from glob import glob
from collections import defaultdict
import seaborn as sns
import multiprocessing

sns.set_style('whitegrid')
sns.set_color_codes('colorblind')
sns.set_context('paper', font_scale=1.5)


def main(args):
    kl_values = defaultdict(lambda: [])
    time_series = defaultdict(lambda: {'KL': [], 'time': []})
    time_series_dfs = []

    for folder in args.folder:

        files = glob(os.path.join(folder, '*', 'KL'))
        for file_ in files:
            with open(file_, 'r') as f:
                for line in f.readlines():
                    a, b = line.strip().split(',')
                    kl_values[a].append(float(b))

        files = glob(os.path.join(folder, '*', '*_KLpq.txt'))
        for repeat_number, file_ in enumerate(files):

            sampler_name = file_.split('_KLpq.txt')[0].split('/')[-1]
            opened_df = pd.read_csv(file_, names=['KL', 'time'])
            opened_df['repeat_id'] = repeat_number
            opened_df['sampler_name'] = sampler_name
            time_series[sampler_name]['KL'].append(opened_df['KL'].values.tolist())
            time_series[sampler_name]['time'].append(opened_df['time'].values.tolist())
            time_series_dfs.append(opened_df)


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for sampler_name in kl_values.keys():
        sns.kdeplot(np.array(kl_values[sampler_name]), shade=True, label=sampler_name, ax=ax)
    ax.set_xlim(0)
    ax.legend(fontsize='x-small')
    fig.savefig(os.path.join(args.out_folder, 'KL_KDE.pdf'))

    time_series_dfs = pd.concat(time_series_dfs, ignore_index=True)
    f = plt.figure()
    ax = f.add_subplot(1, 1,1)
    sns.tsplot(time_series_dfs, time='time', unit='repeat_id', condition='sampler_name', value='KL', ax=ax)
    ax.semilogy()
    ax.set_ylim(0)
    ax.legend(fontsize='x-small')
    f.savefig(os.path.join(args.out_folder, 'KL_timeseries.pdf'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Aggregate KLs from experimental runs and creates a histogram')
    parser.add_argument('-f', '--folder', help='Folder with the experimental runs', required=True, nargs='+')
    parser.add_argument('-of', '--out_folder', help='Folder to save plots in', required=True)
    args = parser.parse_args()
    main(args)