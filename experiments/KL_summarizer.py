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

samplers = ['ISSampler', 'ABCSampler', 'MCSampler', 'RVISampler']
colors = sns.color_palette('colorblind', len(samplers))

def custom_time_series(timeseries_files, ax):
    colors = sns.color_palette('colorblind', 4)
    for color, sampler_name in zip(colors, samplers):
        for y, x in zip(timeseries_files[sampler_name]['KL'], timeseries_files[sampler_name]['time']):
            ax.plot(x,y, color=color, alpha=0.1)

        ax.plot(np.mean(timeseries_files[sampler_name]['time'], axis=0),
                np.mean(timeseries_files[sampler_name]['KL'], axis=0), color=color)

    from matplotlib.lines import Line2D
    custom_lines = []
    for color in colors:
        custom_lines.append(Line2D([0], [0], color=color, linestyle='-'))
    ax.legend(custom_lines, ['ISSampler', 'ABCSampler', 'MCSampler', 'RVISampler'])
    ax.semilogy()

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
    for color, sampler_name in zip(colors, samplers):
        sns.kdeplot(np.array(kl_values[sampler_name]), shade=True, label=sampler_name, ax=ax, color=color)
    ax.set_xlabel('KL')
    ax.set_ylabel('Density')
    ax.set_xlim(0)
    ax.legend(fontsize='x-small')
    fig.savefig(os.path.join(args.out_folder, args.name+'KL_KDE.pdf'))

    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    custom_time_series(time_series, ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('KL')
    ax.set_ylim(0)
    f.savefig(os.path.join(args.out_folder, args.name+'KL_timeseries_custom.pdf'))

    time_series_dfs = pd.concat(time_series_dfs, ignore_index=True)
    f = plt.figure()
    ax = f.add_subplot(1, 1,1)
    sns.tsplot(time_series_dfs, time='time', unit='repeat_id', condition='sampler_name', value='KL', ax=ax)
    ax.semilogy()
    ax.set_ylim(0)
    ax.legend(fontsize='x-small')
    f.savefig(os.path.join(args.out_folder, args.name+'KL_timeseries.pdf'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Aggregate KLs from experimental runs and creates a histogram')
    parser.add_argument('-f', '--folder', help='Folder with the experimental runs', required=True, nargs='+')
    parser.add_argument('-of', '--out_folder', help='Folder to save plots in', required=True)
    parser.add_argument('-n', '--name', help='name to append', required=False, default='summarized_')
    args = parser.parse_args()
    main(args)