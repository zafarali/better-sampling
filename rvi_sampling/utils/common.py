import random
import numpy as np
import torch
import logging

EPSILON = 0

def set_global_seeds(seed):
    # sets global seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_device(use_cuda):
    device_cpu = torch.device("cpu")
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            logging.warning("GPU not available. Reverting to CPU...")
            device = device_cpu
    else:
        device = device_cpu

    return device