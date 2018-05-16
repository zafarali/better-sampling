import random
import numpy as np
import torch

EPSILON = 1e-10

def set_global_seeds(seed):
    # sets global seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
