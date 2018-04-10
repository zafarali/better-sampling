import os
import numpy as np
import torch
NO_RETURN = 'NORETURN'

class Diagnostic(object):
    _handle = None
    def __init__(self, frequency, rounding=2):
        """
        A diagnostic object that is called every `frequency` steps
        :param frequency:
        """
        self.frequency = frequency
        self.count = 0
        self.rounding = rounding


    def set_sampler_name(self, sampler_name):
        self.sampler_name = sampler_name

    def _call(self, results, other_information=None):
        raise NotImplementedError('Diagnostic._call is not implemented')

    def __call__(self, results, other_information=None):
        self.count += 1
        if self.count % self.frequency == 0:
            return self._call(results, other_information)
        else:
            return NO_RETURN


    def format_string(self, to_format):
        return to_format

    def format_float(self, to_format):
        if self.rounding:
            return round(to_format, self.rounding)
        else:
            return to_format

class KLDivergenceDiagnostic(Diagnostic):
    _handle = 'KL(p||q)'
    def __init__(self, kl_function, histbin_range, frequency):
        """
        Returns the KL divergence for using the estimated posterior to encode the true posterior

        :param kl_function:
        :param histbin_range:
        :param frequency:
        """
        self.kl_function = kl_function
        self.histbin_range = histbin_range
        super().__init__(frequency, rounding=None)

    def _call(self, results, other_information=None):
        empirical_distribution = results.empirical_distribution(self.histbin_range)
        KL_true_est, _ = self.kl_function(empirical_distribution)
        return self.format_float(KL_true_est)

class ProportionSuccessDiagnostic(Diagnostic):
    """
    Returns the proportion of trajectories that were sucessful so far
    """
    _handle = 'prop'
    def _call(self, results, other_information=None):
        return self.format_float(len(results.trajectories()) / len(results.all_trajectories()))


class DiagnosticHandler(Diagnostic):
    """
    A "meta-diagnositc" that calls a whole bunch of other diagnostic
    Basically this is a list for `Diagnostic` objects.
    """
    def __init__(self, diagnostics, sampler_name=None, verbose=False):
        """
        :param diagnostics: A list of diagnostics
        :param sampler_name: the name of the sampler
        :param verbose:
        """
        super().__init__(1) # logging frequency is 1, by default `DiagnosticHandler`s should be called every step
        self.sampler_name = sampler_name
        self.diagnostics = diagnostics
        self.diagnostic_information = []
        self.verbose = verbose

    def _call(self, results, other_information=None):
        tmpstr, _= self._call_all_diagnostics(results, other_information)
        return tmpstr

    def _call_all_diagnostics(self, results, other_information=None, optional_fn=None):
        """
        Calls all the diagnostics that are being handled.
        :param results: results to be processed
        :param other_information: other information to be used
        :param optional_fn: optional printing functions, is called with the diagnostic name and the information
                            returned from it. Useful for printing or saving. (See SimpleFileHandler
                            and TensorBoardHandler)
        :return:
        """
        tmpstr = '{}: '.format(self.count)
        diagnostics_this_iter = [self.count]
        diagnostic_count = 0
        for diagnostic in self.diagnostics:
            diagnosed = diagnostic(results, other_information)
            if diagnosed == NO_RETURN:
                continue
            diagnostic_count += 1
            diagnostics_this_iter.append((diagnostic._handle, diagnosed))
            tmpstr += '{}={}, '.format(*diagnostics_this_iter[-1])
            if optional_fn is not None:
                optional_fn(*diagnostics_this_iter[-1], self.count)
        if self.verbose: print(tmpstr)
        if diagnostic_count == 0:
            return NO_RETURN, NO_RETURN
        self.diagnostic_information.append(diagnostics_this_iter)
        return tmpstr, diagnostics_this_iter


class TensorBoardHandler(DiagnosticHandler):
    """
    TensorBoardHandler
    """
    def __init__(self, diagnostics, log_dir=None, sampler_name='', verbose=False):
        super().__init__(diagnostics, sampler_name, verbose)
        self.initialized = False
        self.log_dir = log_dir

    def initialize(self):
        # extra abstraction so that it can work in the multiprocessing case.
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise Exception('You do not have tensorboardX installed. Therefore you cannot use this.')
        self.writer = SummaryWriter(self.log_dir, self.sampler_name)

        def log_to_tensorboard(text, to_log, count):
            if isinstance(to_log, (np.ndarray, torch.Tensor, list)):
                self.writer.add_histogram(text, to_log, count)
            else:
                self.writer.add_scalar(text, to_log, count)

        self.log_to_tensorboard = log_to_tensorboard
        self.initialized = True

    def _call(self, results, other_information=None):
        if not self.initialized: self.initialize()
        tmpstr, _  = self._call_all_diagnostics(results, other_information, optional_fn=self.log_to_tensorboard)
        return tmpstr

class FileSaverHandler(DiagnosticHandler):
    """
    Something like TensorBoardHandler, but without all the heavy weight stuff
    of saving into TensorBoard. Easier to export data from to do downstream analysis.
    """
    def __init__(self, diagnostics, log_dir, sampler_name, verbose=False):
        super().__init__(diagnostics, sampler_name, verbose)
        self.initialized = False
        self.log_dir = log_dir
        self.sampler_name = sampler_name

    def initialize(self):
        # we use this special initialization file so that it is only initialized
        # once this is within a multiprocessing loop
        self.initialized = True
        def log_to_file(text, to_log, count):
            text = text.replace(')', '').replace('(', '').replace('|', '') # clean the string for backward compatability
            with open(os.path.join(self.log_dir, self.sampler_name+'_'+text+'.txt'), 'a') as f:
                f.write(str(to_log)+','+str(count)+'\n')

        self.log_to_file = log_to_file

    def _call(self, results, other_information=None):
        if not self.initialized: self.initialize()
        tmpstr, _ = self._call_all_diagnostics(results, other_information, optional_fn=self.log_to_file)
        return tmpstr

def create_diagnostic(sampler_name, args, folder_name, kl_function=None):
    """
    Creates diagnostics for our experiments
    :param sampler_name: Name of the sampler
    :param args: arguments for the sampler
    :param folder_name: the name of the folder to save things into (only for tensorboard)
    :param kl_function: The KL divergence function to use
    :return:
    """
    diagnostics = [ProportionSuccessDiagnostic(2)]
    if kl_function is not None: diagnostics += [KLDivergenceDiagnostic(kl_function, args.rw_width, 2)]

    if args.no_tensorboard:
        diagnostic_handler = FileSaverHandler(diagnostics, os.path.join(folder_name), sampler_name)
    else:
        print('Tensorboard Logging at: {}'.format(os.path.join(folder_name, sampler_name)))
        diagnostic_handler = TensorBoardHandler(diagnostics,log_dir=os.path.join(folder_name, sampler_name))

    return diagnostic_handler
