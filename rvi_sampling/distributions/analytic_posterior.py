from scipy.misc import comb
from rvi_sampling.utils.common import EPSILON
import numpy as np
import matplotlib.pyplot as plt

class AnalyticPosterior(object):
    def pdf(self, x, d):
        """
        Obtain the posterior probability
        P(x | d)
        :param x: the starting position
        :param d: the ending position
        :return:
        """
        raise NotImplementedError('You must implement this method')


class TwoStepRandomWalkPosterior(AnalyticPosterior):
    """
    The analytic posterior for a two step random walk with step sizes [-1, +1]
    The -1 step is taken with probability p.
    A total of T steps are taken. c denotes the width of the discrete uniform prior

    Example of how to use:
    ```
    c = 15
    p = 0.5
    T = 100
    posterior = TwoStepRandomWalkPosterior(c, p, T)

    # this will plot the posterior when a particle is found at position 10
    plt.scatter(np.arange(-c, c), [posterior.pdf(x, 10) for x in np.arange(-c, c)])
    plt.xlabel(r"$x_0$")
    plt.ylabel('Unnormalized Probability')
    plt.title('Posterior Distribution')
    ```
    """
    def __init__(self, c, p, T):
        """
        :param c: width of the discrete uniform prior centered around 0
        :param p: the probability of taking a step of size -1
        :param T: the total number of steps taken
        """
        self.c = c
        self.p = p
        self.T = T-1 # T steps is T-1 transitions
        self.support = np.arange(-self.c, self.c+1)

    def pdf(self, x, d):
        """
        Obtain the posterior for the two step random walk
        :param x: the starting position
        :param d: the ending position
        :return: 0 <= float <= 1
        """
        c, T, p = self.c, self.T, self.p
        term = 0.5 * (d-x+T)
        if not (x in range(-c, c+1, 1) and term in range(0, T+1)):
            return 0
        else:
            return (1/(2*c))*comb(T, term)*(p**term)*((1-p)**(T-term))


    def plot(self, observed_d, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        prior_domain = np.arange(-self.c-2, self.c+2)
        pdf = np.array([self.pdf(x, observed_d) for x in prior_domain])
        pdf /= pdf.sum()
        ax.scatter(prior_domain, pdf, **kwargs)
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel('Probability')
        ax.set_title('Posterior Distribution')
        # ax.set_xticklabels(prior_domain)
        return ax

    def expectation(self, observed_d):
        pdf = np.array([self.pdf(x, observed_d) for x in np.arange(-self.c, self.c+1)])
        pdf /= pdf.sum()
        return np.sum(np.arange(-self.c, self.c+1) * pdf)

    def kl_divergence(self, estimated_dist, observed_d, verbose=False):
        KL_true_est = 0
        KL_est_true = 0

        analytic_probs = np.array([float(self.pdf(i, observed_d)) for i in estimated_dist.keys()])
        analytic_probs /= analytic_probs.sum()
        analytic_probs = dict(zip(estimated_dist.keys(), analytic_probs))

        # check if the same values are there in the support
        support_check = [set(analytic_probs.keys()) == set(estimated_dist.keys())]

        # KL is only defined when p>0 and q>0 or both zero
        support_check += [ (analytic_probs[k]>0 and estimated_dist[k]>0) or (analytic_probs[k]==0 and estimated_dist[k]==0) for k in analytic_probs.keys() ]

        if verbose: print('analytic:',analytic_probs)
        if verbose: print('estimated',estimated_dist)
        if verbose: print('support check',support_check)
        if not all(support_check):
            return (np.nan, np.nan)

        for k in estimated_dist.keys():
            value = analytic_probs[k]*np.log(EPSILON + analytic_probs[k]/(EPSILON + estimated_dist[k]))
            if not np.isnan(value):
                KL_true_est += value
            value = estimated_dist[k]*np.log(EPSILON + estimated_dist[k]/(EPSILON + analytic_probs[k]))
            if not np.isnan(value):
                KL_est_true += value

        return KL_true_est, KL_est_true


class MultiWindowTwoStepRandomwWalkPosterior(TwoStepRandomWalkPosterior):
    def __init__(self, window_ranges, p, T):
        """
        A multiwindow two step random walk posterior.
        :param window_ranges: the boundaries of each window
        :param p: the step probabilities
        :param T: the length  of time the stochastic process has been run
        """
        self.window_ranges = window_ranges
        self.support = []
        for (l,r) in self.window_ranges:
            self.support.extend(range(l, r+1))
        self.T = T - 1 # number of transitions
        self.p = p


    def pdf(self, x, d):
        """
        Obtain the posterior for the two step random walk
        :param x: the starting position
        :param d: the ending position
        :return: 0 <= float <= 1
        """
        T, p = self.T, self.p
        c = len(self.support)
        term = 0.5 * (d-x+T)
        if not (x in self.support and term in range(0, T+1)):
            return 0
        else:
            return (1/(2*c))*comb(T, term)*(p**term)*((1-p)**(T-term))

    def plot(self, observed_d, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        pdf = np.array([self.pdf(x, observed_d) for x in self.support])
        pdf /= pdf.sum()
        ax.scatter(self.support, pdf, **kwargs)
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel('Probability')
        ax.set_title('Posterior Distribution')
        return ax

    def expectation(self, observed_d):
        np.array([self.pdf(x, observed_d) for x in self.support])
        pdf /= pdf.sum()
        return np.sum(np.array(self.support) * pdf)


class MultiDimensionalRandomWalkPosterior(AnalyticPosterior):
    def __init__(self, c, p, T, dimensions=2):
        """
        See TwoStepRandomWalkPosterior for details on the arguments.
        """
        self._posteriors = [
            TwoStepRandomWalkPosterior(c, p, T) for  _ in range(dimensions)
        ]
        # TODO(zaf): Merge into TwoStepRandomWalkPosterior?
        self.c = c
        self.p = p
        self.T = T-1
        self.support = np.arange(-self.c, self.c+1)

    def pdf(self, x, d):
        probability = 1.0
        for i, posterior in enumerate(self._posteriors):
            probability *= posterior.pdf(x[i], d[i])

        return probability

    def kl_divergence(self, estimated_dist, observed_d, verbose=False):
        kl_divergences = []
        for i, posterior in enumerate(self._posteriors):
            kl_divergences.append(
                posterior.kl_divergence(
                    estimated_dist[i],
                    observed_d[i],
                    verbose)
                )
        # Return the worst KL Divergence.
        return list(map(np.max, zip(*kl_divergences)))


