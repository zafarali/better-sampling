from .base import StochasticProcess, StochasticProcessOverError
from collections import namedtuple
import numpy as np

SIRParameters = namedtuple('SIRParameters',
                           'population_size infection_rate recovery_rate T delta_t')

class SIR(StochasticProcess):
    """
    The Susceptible-Infected-Recovered Epidemic model

    See Section 3.3.3 of
    Allen, Linda JS. "An introduction to stochastic epidemic models."
    Mathematical epidemiology. Springer, Berlin, Heidelberg, 2008. 81-130.

    Implemented here is a simpler version from Linda Allens slides
    https://drive.google.com/file/d/0BwbIZeNSn5cdUzZlRzdUeUlRVWc/view?usp=sharing
    """

    state_space = 2 # Susceptible, Infected
    action_space = 3
    _SUSCEPTIBLE_DIM = 0 # the dimension of the susceptibles
    _INFECTED_DIM = 1 # the dimension of the infecteds
    def __init__(self,
                 population_size,
                 infection_rate,
                 recovery_rate,
                 prior,
                 T,
                 delta_t=0.01,
                 n_agents=1,
                 seed=1):
        """

        :param population_size: the total size of the population
        :param infection_rate: the rate of infection
        :param recovery_rate: the rate of  recovery
        :param prior: the prior over starting states.
        :param T: the total number of timesteps
        :param delta_t: a small interval to ensure values resemble probabilities
        :param n_agents: he number of agents to simulate at a time
        :param seed:
        """
        super().__init__(seed, T)
        self.population_size = population_size

        # the possible transitions we can take
        self.step_sizes = np.array([
            [-1, 1],
            [0, -1],
            [0, 0]
        ])
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.prior = prior
        self.n_agents = n_agents
        self.delta_t = delta_t
        assert prior.dimensions == self.state_space
        self.dimensions = self.state_space
        self.new_task()

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


        # all_probs = np.stack((prob_new_infection, prob_recovery, prob_resusceptible, prob_new_susceptible, prob_no_change))
        all_probs = np.stack((prob_new_infection, prob_recovery, prob_no_change))
        # normalization here. If these do not sum to 1, then somewhere down the line an error
        # will occur.
        all_probs = all_probs.reshape(inferred_n_agents, -1)
        return all_probs / all_probs.sum(axis=1).reshape(-1, 1)


    def simulate(self, rng=None):
        """
        Simulates a new stochastic process for self.T steps.
        :param rng: a random number generator
        :return: the generated trajectory
        """
        if rng is None:
            rng = self.rng

        x0 = self.prior.rvs()
        trajectories = [x0]
        for i in range(self.T-1):
            x_tm1 = trajectories[-1]
            probability_distribution = self.transition_prob(np.array([x_tm1])).reshape(-1)
            selected_transition = self.step_sizes[np.argmax(self.rng.multinomial(1, probability_distribution))]
            x_t = x_tm1 + selected_transition

            trajectories.append(x_t)

        trajectories = np.array(trajectories)
        assert np.all(trajectories >= 0), 'something went wrong, trajectory must be all non-negative!'
        return trajectories

    def reset_agent_locations(self):
        """
        Reset the location of all agents that are interacting with the stochastic process
        :return: None
        """
        self.transitions_left = self.T - 1
        self.x_agent = np.repeat(self.xT.reshape(1, self.dimensions), self.n_agents, axis=0)

    def reset(self):
        """
        This will reset the game and return the location of the agents.
        :return:
        """
        self.reset_agent_locations()
        return self.x_agent

    def new_task(self):
        """
        Creates a new realization of the stochastic process for agents to interact with
        :return:
        """
        self.true_trajectory = self.simulate()
        self.x0 = self.true_trajectory[0]
        self.xT = self.true_trajectory[-1]
        return self.reset()

    def step(self, actions, reverse=False):
        """
        Allows agents to execute actions within a stochastic process
        :param actions: the numpy array of actions to be executed by each agent
        :param reverse: defines if the step is to be reversed or not
        :return: (current_state,
                  reward (! to be modified by you),
                  done,
                  information about the step)
        """
        if self.transitions_left == 0:
            raise StochasticProcessOverError('You have already reached the end of the episode. Use reset()')

        steps_taken = np.take(self.step_sizes, actions.ravel(), axis=0)

        # note that here we have to do some manipulation with the transition probs
        # since each agent has a custom transition prob

        transition_probs = self.transition_prob(self.x_agent)
        assert np.all(transition_probs>=0)
        step_probs = np.take(transition_probs,
                                    actions.ravel(), axis=1).reshape(self.n_agents, -1)

        step_log_probs = np.log(step_probs)
        reversal_param = -1 if reverse else +1
        self.x_agent = self.x_agent + (steps_taken * reversal_param)
        self.transitions_left -=1
        if self.transitions_left == 0:
            # add the "final reward"
            print('adding the log prob of final!!')
            step_log_probs += np.log(self.prior.pdf(self.x_agent))

        # TODO: we can also probably end the trajectory early
        # if we go over or under the population size.
        if np.any(np.isnan(step_log_probs)): print(step_log_probs, step_probs, transition_probs, self.x_agent, actions, steps_taken)
        return (self.x_agent, step_log_probs.reshape(-1, 1), self.transitions_left == 0, {})

