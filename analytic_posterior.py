from scipy.misc import comb
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
        self.T = T

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
