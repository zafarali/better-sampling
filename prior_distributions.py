import numpy as np

def __discrete_unif_pdf(x, start, n_numbers):
    if x >= start and x <= start +n_numbers:
        return 1/n_numbers
    else:
        return 0

_discrete_unif_pdf = np.vectorize(__discrete_unif_pdf)

class DiscreteUniform(object):
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
        self.dimensions = dimensions
        self.start = start
        self.n_numbers = n_numbers
        self.rng = np.random.RandomState(seed)
        self.support = np.arange(start, start+n_numbers)


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

