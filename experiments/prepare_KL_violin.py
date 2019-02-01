"""
A utility to plot a histogram of KL divergences as a nice plot
this is useful for showing performance of a method on average.

"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import os
import argparse
import pandas as pd
import numpy as np
from glob import glob

def get_hyperparameter(path, hyperparameter_name, splitter='/'):
    # Get the hyperparameter from the path.
    return path.split(hyperparameter_name)[1].split(splitter)[0]

def extract_data(
    template_file,
    statistic,
    hyperparameters,
    sampler_name='ISSampler',
    return_pd=True):
    """

    """
    glob_path = os.path.join(template_file, 'Seed*', sampler_name + '_' + statistic + '.txt')
    files = glob(glob_path)
    datas = []
#     print(files)
    for file_ in files:
        raw_data = np.loadtxt(file_, delimiter=',')
        data = pd.DataFrame(data=raw_data, columns=[statistic, 'trajectories'])
        for hp in hyperparameters:
            data[hp] = get_hyperparameter(file_, hp)
        datas.append(data)
    if return_pd:
        return pd.concat(datas, ignore_index=True)
    else:
        return datas


def main(args):
    print('Reading data from {}'.format(args.template))
    print('Hyperparameters: {}'.format(args.hyperparameters))
    print('Extracting data...')

    extracted_data = []
    for end_point in args.end_points:
        print('Extracting data for end point: {}'.format(end_point))
        extracted_data.extend(
            extract_data(
                args.template.format(
                    end_point=end_point,
                    **{key: '*' for key in args.hyperparameters}),
                statistic=args.statistic,
                sampler_name=args.sampler_name,
                hyperparameters=['end_point', 'Seed'] + list(args.hyperparameters),
                return_pd=False))

    extracted_data = pd.concat(extracted_data, ignore_index=True)
    extracted_data = extracted_data.apply(pd.to_numeric, errors='ignore',)

    print('Data extracted.')

    if args.dry_run:
        print(extracted_data)

    if args.trajectory_count is None:
        if args.sampler_name == 'RVISampler':
            args.trajectory_count = (99, 499, 999, 1999, 2999, 3999, 4999, 5999, 6999, 7999, 8999, 9999)
        else:
            args.trajectory_count = (100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000)

    print('Summarizing data at: {}'.format(args.trajectory_count))

    grouped_data = []
    for trajectory_count in args.trajectory_count:
        print('Summarizing at trajectory count {}'.format(trajectory_count))
        summary_df = extracted_data[np.isclose(
            extracted_data.trajectories, trajectory_count)]

        grouped_data.append(summary_df)

    combined_df = pd.concat(grouped_data)
    combined_df['sampler'] = args.sampler_name
    combined_df['method'] = args.method_name

    print('Data summarized.')
    if args.dry_run:
        print(combined_df)
    else:
        print('Saving...')
        print(combined_df)
        df = combined_df[[
            'method',
            'end_point',
            'trajectories',
            'Seed',
            args.statistic]]
        if os.path.exists(args.save_file):
            df.to_csv(args.save_file, mode='a', header=False)
        else:
            df.to_csv(args.save_file, header=True)
        print('Saved.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Aggregate experimental runs.')
    parser.add_argument(
        '--template',
        required=True,
        help='Template for the folder with the experimental runs')
    parser.add_argument(
        '--sampler_name',
        required=True,
        help='Name of the sampler.')
    parser.add_argument(
        '--statistic',
        help='Statistic to read',
        required=True)
    parser.add_argument(
        '--end_points',
        nargs='+',
        required=False,
        type=int,
        default=(0, 12, 24, 36, 48),
        help='Endpoints to iterate over.')
    parser.add_argument(
        '--trajectory_count',
        nargs='+',
        required=False,
        type=int,
        default=None,
        help='A list containing the number of trajectories after which to summarize.')
    parser.add_argument(
        '--hyperparameters',
        nargs='+',
        required=True,
        help='Hyperparameters to extract from the folder path.')
    parser.add_argument(
        '--save_file',
        help='File to save to',
        required=False,
        default='aggregated.csv')
    parser.add_argument(
        '--dry_run',
        help='Dry run (only print)',
        required=False,
        action='store_true',
        default=False)
    parser.add_argument(
        '--method_name',
        help='Name of the method',
        required=False,
        default='method') 
    args = parser.parse_args()
    main(args)
