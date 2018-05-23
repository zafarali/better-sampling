import os
import time
# create a new folder:
create_folder = lambda f: [os.makedirs(os.path.join('./', f)) if not os.path.exists(os.path.join('./', f)) else False]


def touch(path):
    """
    Create a new file with title "path"
    :param path:
    :return:
    """
    with open(path, 'a'):
        os.utime(path, None)

def put(path, text):
    """
    Saves text into the path
    (!) the path must not already exist.
    :param path:
    :param text:
    :return:
    """
    with open(path, 'w') as f:
        if type(text) is list:
            f.write('\n'.join(text))
        else:
            f.write(text)

def stash(path, text):
    with open(path, 'a') as f:
        if type(text) is list:
            f.write('\n'.join(text))
        else:
            f.write('\n'+text)


def argparse_saver(path, parser):
    "saves an argparse object"
    kwargs = [ '{},{}'.format(k, v) for k,v in parser._get_kwargs()]
    put(path, kwargs)


def create_folder_name(outfolder, experiment_name, alternative='results'):
    """
    Returns a standardized folder name
    :param experiment_name:
    :param alternative:
    :return:
    """
    if experiment_name != '':
        folder_name = '{}_{}'.format(experiment_name, time.strftime('%a-%d-%m-%Y__%H-%M-%s'))
    else:
        folder_name = '{}_{}'.format(alternative, time.strftime('%a-%d-%m-%Y__%H-%M-%s'))

    return os.path.join(outfolder, folder_name)

