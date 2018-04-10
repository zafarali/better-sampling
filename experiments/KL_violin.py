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

def main(args):
    kl_values = defaultdict(lambda: [])
    time_series = defaultdict(lambda: {'KL': [], 'time': []})
    dfs = []

    for folder in args.folder:
        files = glob(os.path.join(folder, '*', 'KL'))
        print('Number of files: {}'.format(len(files)))
        nn_size, reward_clip = folder.split('/')[-1].replace('hps_', '').split('-')
        # holds information about this run. We can do a local violin plot on these values.
        # or we can put them into a df.
        for file_ in files:
            with open(file_, 'r') as f:
                for line in f.readlines():
                    a, b = line.strip().split(',')
                    kl_values[a].append(np.log10(float(b)))
        this_df = pd.DataFrame(kl_values)
        this_df['reward_clip'] = -float(reward_clip)
        this_df['nn'] = nn_size
        dfs.append(this_df)
        # print(dfs[-1])
        # files = glob(os.path.join(folder, '*', '*_KLpq.txt'))
        # for repeat_number, file_ in enumerate(files):
        #     sampler_name = file_.split('_KLpq.txt')[0].split('/')[-1]
        #     opened_df = pd.read_csv(file_, names=['KL', 'time'])
        #     opened_df['repeat_id'] = repeat_number
        #     opened_df['sampler_name'] = sampler_name
        #     time_series[sampler_name]['KL'].append(opened_df['KL'].values.tolist())
        #     time_series[sampler_name]['time'].append(opened_df['time'].values.tolist())
        #     time_series_dfs.append(opened_df)
    merged_df = pd.concat(dfs)
    melted_df = pd.melt(merged_df, id_vars=['nn', 'reward_clip'], value_vars=samplers,
                        var_name='Sampler', value_name='KL')
    print(melted_df.sample(100))
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax = sns.factorplot(x="Sampler", y="KL",
                        data=melted_df[(melted_df['Sampler'] != 'RVISampler') & (melted_df['reward_clip'] != -10)], ax=ax, kind='violin', palette=dict(zip(samplers, colors)))
    ax.set_ylim(-4, -0.5)
    ax = fig.add_subplot(122)
    # ax = sns.stripplot(x="reward_clip", y="KL", hue="Sampler", data=melted_df, ax=ax)
    _ = sns.factorplot(x="reward_clip", y="KL", hue="Sampler",
                        data=melted_df[(melted_df['Sampler'] == 'RVISampler')], ax=ax, kind='violin', palette=dict(zip(samplers, colors)))
    ax.set_ylim(-4, -0.5)
    fig.savefig('./TEST_KLVIOLIN.pdf')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Aggregate KLs from experimental runs and creates a histogram')
    parser.add_argument('-f', '--folder', help='Folder with the experimental runs', required=True, nargs='+')
    parser.add_argument('-of', '--out_folder', help='Folder to save plots in', required=True)
    parser.add_argument('-n', '--name', help='name to append', required=False, default='summarized_')
    args = parser.parse_args()
    main(args)