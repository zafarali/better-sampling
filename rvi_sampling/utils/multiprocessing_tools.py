from ..samplers.RVISampler import RVISampler
from ..StochasticProcess import PyTorchWrap

def run_sampler(args):
    """
    Used to call samplers in a multiprocessing `map` call
    :param args:
    :return:
    """
    sampler, rw, MC_samples = args
    if isinstance(sampler, RVISampler):
        return sampler.solve(PyTorchWrap(rw), MC_samples)
    else:
        return sampler.solve(rw, MC_samples)
