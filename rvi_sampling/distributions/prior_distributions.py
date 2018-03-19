import numpy as np

def __discrete_unif_pdf(x, start, n_numbers):
    # TODO ensure that only ints are passed here.
    # TODO this should return 0 for any non integer number.
    if x >= start and x <= start +n_numbers:
        return 1/n_numbers
    else:
        return 0

_discrete_unif_pdf = np.vectorize(__discrete_unif_pdf)

class AbstractPriorDistribution(object):
    support = None
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def rvs(self):
        """Draw a random variable"""
        raise NotImplementedError

    def draw(self):
        return self.rvs()

    def pdf(self, x):
        """The probability of observing x"""
        raise NotImplementedError

class DiscreteUniform(AbstractPriorDistribution):
    """
    A discrete uniform distribution mimic-ing some of
    scipy.stats.uniform methods.
    """
    def __init__(self, dimensions, start, n_numbers, seed=2017):
        """
        Creates a distribution
        :param dimensions: The number of dimensions/state space
        :param start: The start point of the range of allowable numbers
        :param n_numbers: The number of allowable numbers
        """
        super().__init__(seed)
        self.dimensions = dimensions
        self.start = start
        self.n_numbers = n_numbers
        self.support = np.arange(start, start+n_numbers+1)


    def rvs(self):
        return np.array([self.rng.choice(self.support) for i in range(self.dimensions)])

    def draw(self):
        return self.rvs()

    def pdf(self, x):
        probs = _discrete_unif_pdf(x, self.start, self.n_numbers)
        if len(x.shape) == 1:
            return probs.prod()
        else:
            return probs.prod(axis=1)


class MultiWindowDiscreteUniform(AbstractPriorDistribution):
    def __init__(self, dimensions, window_ranges=[(-5, 5)], seed=2017):
        super().__init__(seed)
        self.dimensions = dimensions
        self.window_ranges = window_ranges
        self.support = []
        for (l,r) in self.window_ranges:
            self.support.extend(range(l, r+1))

        # for backward compatability
        # TODO: find a way around this.
        self.start = np.min(self.support)
        self.n_numbers = len(self.support)

    def rvs(self):
        return np.array([self.rng.choice(self.support) for _ in range(self.dimensions)])

    def draw(self):
        return self.rvs()

    def pdf(self, x):
        probs = np.isin(x, self.support)/len(self.support)
        if len(x.shape) == 1:
            return probs.prod()
        else:
            return probs.prod(axis=1)

