"""
A utility to plot a histogram of KL divergences as a nice plot
this is useful for showing performance of a method on average.

Example of possible commands:
python KL_violin.py -f './April10/hps_*' -of './April10'  -hue nn -t 'RVI Performance on One Window RW'
python KL_violin.py -f './April10/twhps/twhps_*' -of './April10'  -hue nn -t 'RVI Performance on Two Window RW'
python KL_violin.py -f './April10/twhps/twhps_161616-10' -of './April10' -hue Sampler -t 'Baseline Comparison on Two Window RW' -k violin
python KL_violin.py -f './April10/hps_323232-10' -of './April10' -hue Sampler -t 'Baseline Comparison on One Window RW' -k violin


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
sns.set_context('poster', font_scale=1)

samplers = ['ISSampler', 'ABCSampler', 'MCSampler', 'RVISampler']


def get_color_map(hue):
    if hue == 'Sampler':
        colors = ["dusty green", "dusty blue", "dusty purple" , "dusty red"]
        colors = sns.xkcd_palette(colors)
        return dict(zip(samplers, colors))
    else:
        # colors = sns.color_palette('colorblind', len(hue))
        sns.set_palette('colorblind')
        return None
# colors = sns.color_palette('colorblind', len(samplers)+2)

def main(args):
    time_series = defaultdict(lambda: {'KL': [], 'time': []})
    dfs = []

    for folder in args.folder:
        files = glob(os.path.join(folder, '*', 'KL'))
        print('Number of files: {}'.format(len(files)))
        # holds information about this run. We can do a local violin plot on these values.
        # or we can put them into a df.
        for file_ in files:
            # print(file_)
            kl_values = defaultdict(lambda: [])
            nn_size, reward_clip = file_.split('hps_')[-1].split('/')[0].split('-')
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
    # print(melted_df.sample(100))
    fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax1 = sns.factorplot(x="Sampler", y="KL",
    #                     data=melted_df[(melted_df['Sampler'] != 'RVISampler') & (melted_df['reward_clip'] != -10)],
    #                     ax=ax1, kind='violin', palette=dict(zip(samplers, colors)))
    # do not plot others for now because we want this graph to focus on RVI changes.
    ax2 = fig.add_subplot(111)
    # ax = sns.stripplot(x="reward_clip", y="KL", hue="Sampler", data=melted_df, ax=ax)
    if args.only_rvi:
        melted_df = melted_df[(melted_df['Sampler'] == 'RVISampler')]
    _ = sns.factorplot(x="reward_clip", y="KL", hue=args.hue, kind=args.kind,
                    data=melted_df, ax=ax2, palette=get_color_map(args.hue), lw=0.5, legend_out=True)

    ax2.legend(fontsize='x-small', frameon=True, framealpha=0.4)
    if args.only_rvi or args.no_legend: ax2.legend_.remove()
    ax2.set_title(args.title)
    ax2.set_ylabel(r'$\log_{10}$(KL(true|estimated))')
    # ax2.plot(np.arange(0, 4), np.arange(0, 4))
    if args.hue == 'Samplers': ax2.set(ylim=(-3, -1.5)) # use this funky syntax because this is not a matplotlib axis anymore
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_folder, './KL_plots_{}.pdf'.format(args.title.replace(' ', '_'))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Aggregate KLs from experimental runs and creates a histogram')
    parser.add_argument('-f', '--folder', help='Folder with the experimental runs', required=True, nargs='+')
    parser.add_argument('-of', '--out_folder', help='Folder to save plots in', required=True)
    parser.add_argument('-t', '--title', help='title of plot', required=False, default='')
    parser.add_argument('-kind', '--kind', help='kind of plot (can only be violin)', required=False, default=None)
    parser.add_argument('-rvi', '--only_rvi', help='Only plot RVI', required=False, default=False, action='store_true')
    parser.add_argument('-hue', '--hue', help='The hue to use (Sampler or nn)', required=False, default='nn')
    parser.add_argument('-nolegend', '--no_legend', help='Removes the legend', required=False, default=False, action='store_true')
    args = parser.parse_args()
    main(args)
    