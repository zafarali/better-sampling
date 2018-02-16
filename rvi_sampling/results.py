import os
import numpy as np
import matplotlib.pyplot as plt
from .plotting import plot_trajectory_time_evolution, plot_mean_trajectories

class SamplingResults(object):
    """
    An abstraction on the results obtained
    """
    def __init__(self, sampler_name, true_trajectory, histbin_range=None):
        self.sampler_name = sampler_name
        self.true_trajectory = true_trajectory
        self._all_trajectories = None
        self._trajectories = None
        self._posterior_particles = None
        self._posterior_weights = None
        self._histbin_range = histbin_range

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
        if posterior_particles is not None:
            assert self._posterior_particles is None
            self._posterior_particles = posterior_particles
            return self._posterior_particles
        else:
            self._posterior_particles = posterior_particles

    def posterior_weights(self, posterior_weights=None):
        """
        Used to weight the respective values in the posterior
        :param posterior_weights:
        :return:
        """
        if posterior_weights is not None:
            assert self._posterior_weights is None
            self._posterior_weights = posterior_weights
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
        """
        The variance of the posterior
        # from https://statweb.stanford.edu/~owen/mc/Ch-var-is.pdf
        :param weighted:
        :return:
        """
        posterior_particles = np.array(self._posterior_particles).reshape(-1)
        posterior_weights = np.array(self._posterior_weights).reshape(-1)

        expected_value = self.expectation(weighted)

        if weighted:
            modified_weights = (posterior_weights / posterior_weights.sum())**2
            terms = (posterior_particles - expected_value)**2
            return np.sum(modified_weights*terms)
        else:
            return np.mean((posterior_weights*posterior_particles - expected_value)**2)

    def plot_distribution(self, histbin_range=None, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        _histbin_range = self._histbin_range if not histbin_range else histbin_range

        ax.hist(np.array(self._posterior_particles).reshape(-1),
                normed=True,
                bins=np.arange(-_histbin_range-1, _histbin_range+2)+0.5,
                weights=np.array(self._posterior_weights).reshape(-1),
                **kwargs)
        ax.set_xlabel('x_0')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram of trajectory starting positions')
        return ax

    def plot_trajectory_evolution(self, dimension=0, step=5, ax=None):
        return plot_trajectory_time_evolution(self.trajectories(), dimension, step=step, ax=ax)

    def plot_all_trajectory_evolution(self, dimension=0, step=20, ax=None):
        return plot_trajectory_time_evolution(self.all_trajectories(), dimension, step=step, ax=ax)

    def plot_mean_trajectory(self, label=None, ax=None):
        trajectories = self.trajectories()
        ts = np.arange(len(trajectories[0]))
        if label is None:
            label = self.sampler_name

        return plot_mean_trajectories(trajectories, ts, self.true_trajectory, ax=ax)

    def plot_mean_all_trajectory(self, label=None, ax=None):
        trajectories = self.all_trajectories()
        ts = np.arange(len(trajectories[0]))
        if label is None:
            label = self.sampler_name

        return plot_mean_trajectories(trajectories, ts, self.true_trajectory, ax=ax)

    def save_results(self, path):
        results_dict = dict(sampler_name=self.sampler_name,
                            true_trajectory=self.true_trajectory.tolist(),
                            all_trejctories=[traj.tolist() for traj in self._all_trajectories],
                            trajectories=[traj.tolist() for traj in self._trajectories],
                            posterior_particles= [float(p) for p in self._posterior_particles] if self._posterior_particles is not None else None,
                            posterior_weights= [ float(w) for w in self._posterior_particles] if self._posterior_weights is not None else None)

        import json
        with open(os.path.join(path, 'trajectory_results_{}'.format(self.sampler_name)), 'w') as f:
            json.dump(results_dict, f)

    def prop_success(self):
        return len(self.trajectories())/len(self.all_trajectories())

    def summary_builder(self):
        return 'Start Estimate: {:3g}, Variance: {:3g}, Prop Success: {:3g}'.format(self.expectation(), self.variance(), self.prop_success())

    def summary(self):
        template_string = '\n'
        template_string += '*' * 45
        template_string += '\nSampler: {}\n'.format(self.sampler_name)
        template_string += str(self.summary_builder()) +'\n'
        template_string += '*'*45
        template_string += '\n'

        return template_string

class ImportanceSamplingResults(SamplingResults):
    def effective_sample_size(self):
        """
        A diagnostic for the quality of the importance sampling scheme.
        The variance in the estimate of the expectation is equal to
        that if we had done `effective_sample_size` number of monte carlo
        estimates.
        :return:
        """
        posterior_weights = np.array(self._posterior_weights).reshape(-1)
        denominator = np.sum(posterior_weights**2)
        numerator = np.sum(posterior_weights)**2
        return numerator/denominator

    def variance(self, weighted=True):
        return super().variance(weighted)

    def expectation(self, weighted=True):
        return super().expectation(weighted)

    def summary_builder(self):
        template = super().summary_builder()
        ess_string = ' ESS: {:3g}'.format(self.effective_sample_size())
        template += ess_string
        return template

class RLSamplingResults(ImportanceSamplingResults):
    pass