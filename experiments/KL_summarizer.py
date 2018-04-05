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
import argparse
import numpy as np
from glob import glob
from collections import defaultdict
import seaborn as sns
sns.set_style('whitegrid')
sns.set_color_codes('colorblind')
sns.set_context('paper', font_scale=1.5)


def main(args):
    kl_values = defaultdict(lambda: [])

    for folder in args.folder:
        search_path = os.path.join(folder, '*', 'KL')

        files = glob(search_path)

        for repeat_number, file_ in enumerate(files):

            with open(file_, 'r') as f:
                for line in f.readlines():
                    a, b = line.strip().split(',')
                    kl_values[a].append(float(b))


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    print(kl_values)
    for sampler_name in kl_values.keys():
        sns.kdeplot(np.array(kl_values[sampler_name]), shade=True, label=sampler_name, ax=ax)
    ax.set_xlim(0)
    fig.savefig('saved_KDE.pdf')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Aggregate KLs from experimental runs and creates a histogram')
    parser.add_argument('-f', '--folder', help='Folder with the experimental runs', required=True, nargs='+')
    args = parser.parse_args()
    main(args)