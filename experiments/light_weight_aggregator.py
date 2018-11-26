# Data aggregator
import argparse
import os

from glob import glob
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import pandas as pd

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
    extracted_data = []
    for end_point in args.end_points:
        extracted_data.extend(
            extract_data(
                template.format(
                    end_point=end_point,
                    **{key: '*' for key in args.hyperparameters}),
                statistic=args.statistic,
                sampler_name=args.sampler_name,
                hyperparameters=['end_point', 'n_agents', 'Seed'] + list(args.hyperparameters),
                return_pd=False))

    extracted_data = pd.concat(extracted_data, ignore_index=True)
    extracted_data = extracted_data.apply(pd.to_numeric, errors='ignore',)

    grouped_data = []

    for trajectory_count in args.trajectory_count
        summary_df = extracted_data[np.isclose(extracted_data.trajectories, trajectory_count)].groupby(
            ['end_point']+list(args.hyperparameters))

        tag = '{}_{}_mean'.format(args.statistic, trajectory_count)
        mean_kls = summary_df.mean().swaplevel()
        mean_kls[tag] = mean_kls[args.statistic]
        grouped_data.append(tag)


        tag = '{}_{}_std'.format(args.statistic, trajectory_count)
        std_kls = summary_df.std().swaplevel()
        std_kls[tag] = std_kls[args.statistic]
        grouped_data.append(std_kls[tag])


        counts = extracted_data[np.isclose(extracted_data.trajectories, trajectory_count)].groupby(
            ['end_point']+list(args.hyperparameters)).count().swaplevel()

        counts['count'] = counts[args.statistic]
        counts['proportion'] = counts['count']/counts['Seed']

        grouped_data.append(counts['count'])
        grouped_data.append(counts['proportion'])

    combined_df = pd.concat(grouped_data, axis=1)
    combined_df['sampler'] = args.sampler_name

    if args.dryrun:
        print(combined_df)
    else:
        with open(args.save_file, 'w') as f_:
            f_.write(combined_df.to_csv())


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
        default=(500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000),
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
        '--dryrun',
        help='Dry run (only print)',
        required=False,
        action='store_true',
        default=False)
    
    args = parser.parse_args()
    main(args)
