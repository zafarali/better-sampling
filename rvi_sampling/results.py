import os
import numpy as np
import matplotlib.pyplot as plt
from .utils.plotting import plot_trajectory_time_evolution, plot_mean_trajectories
from .utils.stats_utils import empirical_distribution
class SamplingResults(object):
    """
    An abstraction on the results obtained
    """
    _importance_sampled = False
    def __init__(self, sampler_name, true_trajectory=None, histbin_range=None):
        self.sampler_name = sampler_name
        self.true_trajectory = true_trajectory
        self._all_trajectories = None
        self._trajectories = None
        self._posterior_particles = None
        self._posterior_weights = None
        self._histbin_range = histbin_range

    @classmethod
    def from_information(ResultClass,
                         sampler_name,
                         all_trajectories,
                         trajectories,
                         particles='auto',
                         weights='auto'):
        """
        Quick constructor class for Results
        :param sampler_name: name of sampler
        :param all_trajectories: all trajectories
        :param trajectories: trajectories that were successful
        :param particles: particles in the posterior
        :param weights: weights of each particle.
        :return:
        """
        result = ResultClass(sampler_name)
        result.all_trajectories(all_trajectories)
        result.trajectories(trajectories)
        if weights == 'auto' and particles == 'auto':
           result.create_posterior()
        elif weights == 'auto' or particles == 'auto':
            raise KeyError('Both weights and particles must be auto.')
        else:
            result.posterior_particles(particles)
            result.posterior_weights(weights)

        return result

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

    def posterior_particles(self, posterior_particles=None):
        return self.posterior(posterior_particles)
    def posterior(self, posterior_particles=None):
        """
        Used to set the values in the posterior
        :param posterior_particles:
        :return:
        """
        if posterior_particles is not None:
            assert self._posterior_particles is None
            self._posterior_particles = np.array(posterior_particles)

            # Backward compat:
            if len(self._all_trajectories[0]) == 1:
                self._posterior_particles = self._posterior_particles.reshape(-1)
            return self._posterior_particles
        else:
            return self._posterior_particles

    def posterior_weights(self, posterior_weights=None):
        """
        Used to weight the respective values in the posterior
        :param posterior_weights:
        :return:
        """
        if posterior_weights is not None:
            assert self._posterior_weights is None
            self._posterior_weights = np.array(posterior_weights).reshape(-1)
            return self._posterior_weights
        else:
            return self._posterior_weights

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

        self._posterior_weights = np.array(self._posterior_weights).reshape(-1)
        self._posterior_particles = np.array(self._posterior_particles).reshape(-1)

    def expectation(self, weighted=False):
        """
        The expected value of the posterior.
        :param weighted: Will be weighted by the likelihood ratios
        :return:
        """
        posterior_particles = self.posterior_particles()
        posterior_weights = self.posterior_weights()

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
        posterior_particles = self.posterior_particles()
        posterior_weights = self.posterior_weights()

        expected_value = self.expectation(weighted)

        if weighted:
            modified_weights = (posterior_weights / posterior_weights.sum())**2
            terms = (posterior_particles - expected_value)**2
            return np.sum(modified_weights*terms)
        else:
            return np.mean((posterior_weights*posterior_particles - expected_value)**2)

    def plot_distribution(self, histbin_range=None, ax=None, **kwargs):
        """
        Plots the distribution of the posterior
        :param histbin_range:
        :param ax:
        :param kwargs:
        :return:
        """
        if ax is None:
            ax = plt.gca()

        _histbin_range = self._histbin_range if not histbin_range else histbin_range

        ax.hist(self.posterior_particles(),
                normed=True,
                bins=np.arange(-_histbin_range-2, _histbin_range+2)+0.5,
                weights=self.posterior_weights(),
                **kwargs)
        ax.set_xlabel('x_0')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram of trajectory starting positions')
        return ax

    def plot_trajectory_evolution(self, dimension=0, step=5, ax=None):
        """
        Plots the evolution of accepted trajectories over time
        :param dimension:
        :param step:
        :param ax:
        :return:
        """
        if type(dimension) is list:
            for dim in dimension:
                plot_trajectory_time_evolution(self.trajectories(), dim, step=step, ax=ax)
        else:
            return plot_trajectory_time_evolution(self.trajectories(), dimension, step=step, ax=ax)

    def plot_all_trajectory_evolution(self, dimension=0, step=20, ax=None):
        """
        Plots the evolution all trajectories over time
        :param dimension:
        :param step:
        :param ax:
        :return:
        """
        if type(dimension) is list:
            for dim in dimension:
                plot_trajectory_time_evolution(self.all_trajectories(), dim, step=step, ax=ax)
        else:
            return plot_trajectory_time_evolution(self.all_trajectories(), dimension, step=step, ax=ax)

    def plot_mean_trajectory(self, label=None, ax=None):
        """
        Plots the mean successful trajectory
        :param label:
        :param ax:
        :return:
        """
        trajectories = self.trajectories()
        ts = np.arange(len(trajectories[0]))
        if label is None:
            label = self.sampler_name

        return plot_mean_trajectories(trajectories, ts, self.true_trajectory, ax=ax)

    def plot_mean_all_trajectory(self, label=None, ax=None):
        """
        Plots the mean of all trajectories
        :param label:
        :param ax:
        :return:
        """
        trajectories = self.all_trajectories()
        ts = np.arange(len(trajectories[0]))
        if label is None:
            label = self.sampler_name

        return plot_mean_trajectories(trajectories, ts, self.true_trajectory, ax=ax)

    def save_results(self, path):
        """
        Saves results into a json file
        :param path:
        :return:
        """

        prepared_posterior_particles = None
        if len(self._all_trajectories[0]) == 1:
            if self._posterior_particles is not None:
                prepared_posterior_particles = [ float(p) for p in self._posterior_particles]
        elif len(self._all_trajectories[0]) > 1:
            if self._posterior_particles is not None:
                prepared_posterior_particles = [ p.tolist() for p in self._posterior_particles]

        results_dict = dict(
                sampler_name=self.sampler_name,
                true_trajectory=self.true_trajectory.tolist(),
                all_trejctories=[traj.tolist() for traj in self._all_trajectories],
                trajectories=[traj.tolist() for traj in self._trajectories],
                posterior_particles=prepared_posterior_particles,
                posterior_weights= [ float(w) for w in self._posterior_weights] if self._posterior_weights is not None else None)

        import json
        with open(os.path.join(path, 'trajectory_results_{}'.format(self.sampler_name)), 'w') as f:
            json.dump(results_dict, f)

    def prop_success(self):
        """
        Calculates the proportion of successful trajectories
        :return:
        """
        return len(self.trajectories())/len(self.all_trajectories())

    def summary_statistics(self):
        """
        Returns all summary statistics that are calculatable
        :return:
        """
        return [self.expectation(), self.variance(), self.prop_success()]

    def summary_builder(self):
        """
        Returns a human readable summary of the statistics calculated.
        :return:
        """
        return 'Start Estimate: {:3g}, Variance: {:3g}, Prop Success: {:3g}'.format(self.expectation(), self.variance(), self.prop_success())

    def summary(self, extra=''):
        """
        Pretty print of the human readable summary
        :param extra:
        :return:
        """
        template_string = '\n'
        template_string += '*' * 45
        template_string += '\nSampler: {}\n'.format(self.sampler_name)
        template_string += str(self.summary_builder()) +'\n'
        if extra != '': template_string += '{}\n'.format(extra)
        template_string += '*'*45
        template_string += '\n'

        return template_string

    def summary_title(self):
        """
        Summary for titling plots.
        :return:
        """
        return '{} Mean: {:3g} Var:{:3g}\nProp: {:3g}'.format(self.sampler_name, *self.summary_statistics())

    def empirical_distribution(self, histbin_range=None):
        """
        Returns an empirical distribution
        :param histbin_range:
        :return:
        """
        _histbin_range = self._histbin_range if not histbin_range else histbin_range

        return empirical_distribution(self.posterior_particles(),self. posterior_weights(), histbin_range=_histbin_range)


class ImportanceSamplingResults(SamplingResults):
    _importance_sampled = True
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
        """
        Variance of the posterior weighted by
        importance sampled weights
        :param weighted:
        :return:
        """
        return super().variance(weighted)

    def expectation(self, weighted=True):
        """
        Excpected value of the posterior weighted by
        importance sampled weights
        :param weighted:
        :return:
        """
        return super().expectation(weighted)

    def summary_builder(self):
        template = super().summary_builder()
        ess_string = ' ESS: {:3g}'.format(self.effective_sample_size())
        template += ess_string
        return template

    def summary_statistics(self):
        return super().summary_statistics() + [self.effective_sample_size()]

    def summary_title(self):
        return '{} Mean: {:3g} Var:{:3g}\nProp: {:3g} ESS: {:3g}'.format(self.sampler_name, *self.summary_statistics())

    def plot_posterior_weight_histogram(self, ax=None, **kwargs):
        """
        Plots the histogram of importance sampled weights
        :param ax:
        :param kwargs:
        :return:
        """
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(111)

        weights = np.array(self._posterior_weights).reshape(-1)
        ax.hist(weights, bins=np.linspace(0, 1, 500), **kwargs)
        ax.set_ylabel('Count')
        ax.set_xlabel('Weight')
        ax.set_title('Min: {:3g}, Max: {:3g}, Mean: {:3g}'.format(np.min(weights), np.max(weights), np.mean(weights)))
        return ax


class RLSamplingResults(ImportanceSamplingResults):
    def plot_reward_curves(self, ax):
        ax.plot(self.rewards_per_episode)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward For Episode')

    def plot_loss_curves(self, ax):
        ax.plot(self.loss_per_episode)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')