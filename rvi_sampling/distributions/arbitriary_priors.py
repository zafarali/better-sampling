import numpy as np
from .prior_distributions import AbstractPriorDistribution


class ArbitriaryPrior(AbstractPriorDistribution):
    """
    Given some states [x1, x2, x3, .. xn] this distribution
    picks a random variable according to probabilities
    [p1, p2, p3, ..., pn]
    """
    def __init__(self, starting_states, state_probabilities=None, seed=2017):
        """
        :param starting_states: the possible starting states
        :param state_probabilities: The probabilities of picking each state
                                    If this is none, we will assign uniform
                                    probability to each starting state
        """
        super().__init__(seed)
        self.support = starting_states
        assert len(starting_states) > 0
        if state_probabilities is None:
            self.probs = np.ones(len(starting_states))/len(starting_states)
        else:
            assert len(starting_states) == len(state_probabilities)
            self.probs = state_probabilities

        assert np.allclose(np.sum(self.probs), 1.0), 'probabilities must sum to 1'

        self.dimensions = starting_states[0].shape[0]

    def rvs(self):
        """
        Draw a random value from this distribution
        :return: returns a draw from the distribution
        """
        return self.support[self.rng.choice(np.arange(len(self.support)), p=self.probs)]

    def pdf(self, x):
        """
        returns the probability of observing x
        :param x:
        :return: a 0 < float < 1
        """
        checks = np.sum(self.support == x, axis=1)==self.dimensions
        if np.all(checks ==0):
            return 0
        return self.probs[np.argmax(checks)]
