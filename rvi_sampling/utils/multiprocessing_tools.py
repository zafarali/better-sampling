from ..stochastic_processes.base import PyTorchWrap

import cProfile

class ProfiledRunner:
    """ Wrapper class for calling sampler with profiling enabled

    Attributes:
        returnValue (:RLSamplingResults): the result from calling sampler
    """
    def __init__(self):
        self.returnValue = None

    def profiled_run_wrapper(self, args):
        self.returnValue = run_sampler(args)


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

def run_sampler_with_profiling(args):
    """
    Used to call the profile wrapper - enables profiling of all samplers
    :param args: args to sampler
    :return: return values from sampler
    """
    prof_runner = ProfiledRunner()
    cProfile.runctx('prof_runner.profiled_run_wrapper(args)', globals(), locals(), 'profile-%s.prof' %args[0]._name)

    return prof_runner.returnValue