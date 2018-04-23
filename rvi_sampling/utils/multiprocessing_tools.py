from ..stochastic_processes.base import PyTorchWrap

def run_sampler(args):
    """
    Used to call samplers in a multiprocessing `map` call
    :param args:
    :return:
    """
    sampler, rw, MC_samples = args
    if sampler._name == 'RVISampler':
        return sampler.solve(PyTorchWrap(rw), MC_samples, verbose=True)
    else:
        return sampler.solve(rw, MC_samples)
