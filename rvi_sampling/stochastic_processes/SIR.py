from .base import StochasticProcess
import numpy as np

class SIR(StochasticProcess):
    """
    The Susceptible-Infected-Recovered Epidemic model

    See Section 3.3.3 of
    Allen, Linda JS. "An introduction to stochastic epidemic models."
    Mathematical epidemiology. Springer, Berlin, Heidelberg, 2008. 81-130.

    Implemented here is a simpler version from Linda Allens slides
    https://drive.google.com/file/d/0BwbIZeNSn5cdUzZlRzdUeUlRVWc/view?usp=sharing
    """

    _state_space = 2 # Susceptible, Infected
    _action_space = 6
    _SUSCEPTIBLE_DIM = 0 # the dimension of the susceptibles
    _INFECTED_DIM = 1 # the dimension of the infecteds
    def __init__(self,
                 population_size,
                 infection_rate,
                 recovery_rate,
                 resusceptible_rate,
                 prior,
                 T,
                 delta_t=0.01,
                 n_agents=1,
                 seed=1):
        """

        :param population_size: the total size of the population
        :param infection_rate: the rate of infection
        :param recovery_rate: the rate of  recovery
        :param resusceptible_rate: the rate of resusceptibility
        :param prior: the prior over starting states.
        :param T: the total number of timesteps
        :param delta_t: a small interval to ensure values resemble probabilities
        :param n_agents: he number of agents to simulate at a time
        :param seed:
        """
        super().__init__(seed, T)
        self.population_size = population_size

        # the possible transitions we can take
        self.transition_sizes = np.array([
            [-1, 1],
            [0, -1],
            # [1, -1],
            # [1, 0],
            [0, 0]
        ])
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.resusceptible_rate = resusceptible_rate
        self.prior = prior
        self.n_agents = n_agents
        self.delta_t = delta_t
        assert prior.dimensions == self._state_space

    def transition_prob(self, state_t):
        """
        Returns the transition probabilities to each of the
        transitionary states from the current state
        :param state_t: the current state
        :return:
        """

        inferred_n_agents = state_t.shape[0]
        # new_infection
        prob_new_infection = self.delta_t * self.infection_rate * np.prod(state_t, axis=1) / self.population_size

        # recovery
        prob_recovery = self.delta_t * self.recovery_rate * state_t[:, self._INFECTED_DIM]

        # resusceptibility
        # prob_resusceptible = self.resusceptible_rate * state_t[:, self._INFECTED_DIM]

        # new susceptible
        # prob_new_susceptible = self.resusceptible_rate * (self.population_size - state_t[:, self._SUSCEPTIBLE_DIM] - state_t[:, self._INFECTED_DIM])

        # no change:

        # prob_no_change = 1 - prob_new_infection - prob_new_susceptible - prob_resusceptible - prob_recovery
        prob_no_change = 1 - prob_new_infection - prob_recovery

        # TODO: check if this normalization is necessary?
        # some papers says that the probabilities do not need to sum to 1
        # and in fact they do not necessarily need to be bounded? What does this mean

        # all_probs = np.stack((prob_new_infection, prob_recovery, prob_resusceptible, prob_new_susceptible, prob_no_change))
        all_probs = np.stack((prob_new_infection, prob_recovery, prob_no_change))
        all_probs = all_probs.reshape(inferred_n_agents, -1)
        # return np.exp(all_probs) / np.exp(all_probs).sum(axis=1).reshape(-1, 1)
        return all_probs / all_probs.sum(axis=1).reshape(-1, 1)


    def simulate(self, rng=None):
        if rng is None:
            rng = self.rng

        x0 = self.prior.rvs()
        trajectories = [x0]
        for i in range(self.T-1):
            x_tm1 = trajectories[-1]
            probability_distribution = self.transition_prob(np.array([x_tm1])).reshape(-1)
            selected_transition = self.transition_sizes[np.argmax(self.rng.multinomial(1, probability_distribution))]
            x_t = x_tm1 + selected_transition

            trajectories.append(x_t)

        trajectories = np.array(trajectories)
        assert np.all(trajectories >= 0), 'something went wrong, trajectory must be all non-negative!'
        return trajectories
