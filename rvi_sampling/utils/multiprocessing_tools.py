from ..stochastic_processes.base import PyTorchWrap

import os
import cProfile

class ProfiledRunner(object):
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
    :param args: ( foldername, args to solver)
    :return: return values from solver
    """
    folder_name = args[0]
    solver_args = args[1]
    profile_path = os.path.join(folder_name, 'profile-%s.prof' %solver_args[0]._name)
    prof_runner = ProfiledRunner()
    cProfile.runctx('prof_runner.profiled_run_wrapper(solver_args)', globals(), locals(), profile_path)

    return prof_runner.returnValue