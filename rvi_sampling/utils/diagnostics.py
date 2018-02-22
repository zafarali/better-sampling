class Diagnostic(object):
    def __init__(self, frequency, rounding=2):
        """
        A diagnostic object that is called every `frequency` steps
        :param frequency:
        """
        self.frequency = frequency
        self.count = 0
        self.sampler_name = 'Unknown'
        self.rounding = rounding


    def set_sampler_name(self, sampler_name):
        self.sampler_name = sampler_name

    def _call(self, results, other_information=None):
        raise NotImplementedError('Diagnostic._call is not implemented')

    def __call__(self, results, other_information=None):
        if self.count % self.frequency == 0:
            self._call(results, other_information)
        self.count+=1

    def format_string(self, to_format):
        return to_format

    def format_float(self, to_format):
        return str(round(to_format, self.rounding))

class KLDivergenceDiagnostic(Diagnostic):
    def __init__(self, analytic_result, histbin_range, frequency):
        """
        Returns the KL divergence for using the estimated posterior to encode the true posterior

        :param analytic_result:
        :param histbin_range:
        :param frequency:
        """
        self.analytic_result = analytic_result
        self.histbin_range = histbin_range
        super().__init__(frequency)

    def _call(self, results, other_information=None):
        empirical_distribution = results.empirical_distribution(self.histbin_range)
        KL_true_est, _ = self.analytic_result(empirical_distribution)
        return 'KL(p|q)={}'.format(self.format_float(KL_true_est))

