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
        return round(to_format, self.rounding)

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
        super().__init__(frequency)

    def _call(self, results, other_information=None):
        empirical_distribution = results.empirical_distribution(self.histbin_range)
        KL_true_est, _ = self.kl_function(empirical_distribution)
        return self.format_float(KL_true_est)

class DiagnosticHandler(Diagnostic):
    def __init__(self, diagnostics, sampler_name=None, verbose=False):
        super().__init__(1) # logging frequency is 1
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
        :param optional_fn: optional printing functions
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
    def __init__(self, diagnostics, log_dir=None, sampler_name='', verbose=False):
        super().__init__(diagnostics, sampler_name, verbose)
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(log_dir, sampler_name)

    def _call(self, results, other_information=None):
        tmpstr, _  = self._call_all_diagnostics(results, other_information, optional_fn=self.writer.add_scalar)
        return tmpstr