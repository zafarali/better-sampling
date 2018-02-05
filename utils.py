import numpy as np

class SamplingResults(object):
    """
    An abstraction on the results obtained
    """
    def __init__(self, sampler_name):
        self.sampler_name = sampler_name
        self._all_trajectories = None
        self._trajectories = None
        self._posterior_particles = None
        self._posterior_weights = None

    def all_trajectories(self, trajectories=None):
        """
        Used to set or retrieve the all trajectories
        :param trajectories:
        :return:
        """
        if trajectories is None:
            return self._all_trajectories
        else:
            self._all_trajectories = trajectories

    def trajectories(self, trajectories=None):
        """
        Used to set or retrieve accepted trajectories
        :param trajectories:
        :return:
        """
        if trajectories is None:
            return self._trajectories
        else:
            self._trajectories = trajectories

    def posterior(self, posterior_particles=None):
        """
        Used to set the values in the posterior
        :param posterior_particles:
        :return:
        """
        if posterior_particles is None:
            assert self._posterior_particles is None
            return self._posterior_particles
        else:
            self._posterior_particles = posterior_particles

    def posterior_weights(self, posterior_weights=None):
        """
        Used to weight the respective values in the posterior
        :param posterior_weights:
        :return:
        """
        if posterior_weights is None:
            assert self._posterior_weights is None
            return self._posterior_weights
        else:
            self._posterior_weights = posterior_weights

    def create_posterior(self):
        """
        Automatically creates a posterior if only trajectories() has been set
        :return:
        """
        assert self._trajectories is not None, 'No trajectories to create posterior from'
        assert self._posterior_particles is None and self._posterior_weights is None, 'Posterior already initialized'
        self._posterior_particles = []
        self._posterior_weights = []
        for trajectory in self._trajectories:
            self._posterior_particles.append(trajectory[0])
            self._posterior_weights.append(1)

    def expectation(self, weighted=False):
        """
        The expected value of the posterior.
        :param weighted: Will be weighted by the likelihood ratios
        :return:
        """
        posterior_particles = np.array(self._posterior_particles).reshape(-1)
        posterior_weights = np.array(self._posterior_weights).reshape(-1)


        numerator = np.sum(posterior_particles * posterior_weights)

        if weighted:
            estimate = numerator / np.sum(posterior_weights)
        else:
            estimate = numerator / len(self._posterior_particles)

        return estimate

    def variance(self, weighted=False):
        posterior_particles = np.array(self._posterior_particles).reshape(-1)
        posterior_weights = np.array(self._posterior_weights).reshape(-1)

        expected_value = self.expectation()
        # from https://statweb.stanford.edu/~owen/mc/Ch-var-is.pdf
        if weighted:
            return np.sum((posterior_weights ** 2) * (posterior_particles - expected_value)**2)
        else:
            return np.mean((posterior_weights*posterior_particles - expected_value)**2)

