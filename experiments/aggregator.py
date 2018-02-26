"""
Utility to aggregate different experimental runs
"""
import os
import argparse
import numpy as np
import pandas as pd
from glob import glob


def main(args):
    search_path = os.path.join(args.folder, args.name+'*__*')
    folders = glob(search_path)
    print('Looking in {}'.format(search_path))
    print('Found {} experimental runs'.format(len(folders)))

    df = pd.DataFrame()

    dfs = []
    for folder in folders:
        try:
            KL, args_path = glob(os.path.join(folder, 'KL'))[0], glob(os.path.join(folder, 'args'))[0]
        except Exception as e:
            print('folder: {} has no information saved'.format(folder))
            continue

        data = dict()
        with open(KL, 'r') as f:
            for line in f.readlines():
                a, b = line.strip().split(',')
                data[a+'_KL'] = float(b)

        with open(args_path, 'r') as f:
            for line in f.readlines():
                a, b = line.strip().split(',')
                data[a] = b
        print(data)
        dfs.append(pd.DataFrame(data, index=[0]))


    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(args.folder, args.save))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Aggregate experimental runs')
    parser.add_argument('-f', '--folder', help='Folder with the experimental runs', required=True)
    parser.add_argument('-n', '--name', help='Name appenditure for the experimental runs', required=False, default='')
    parser.add_argument('-s', '--save', help='File to save to', required=False, default='aggregated.csv')

    args = parser.parse_args()
    main(args)