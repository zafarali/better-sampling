import gym
import numpy as np


class StochasticProcess(gym.Env):
    def __init__(self, seed, T):
        super().__init__()
        self.rng = np.random.RandomState(seed)
        self.T = T

    @property
    def state_space(self):
        self._state_space

    @property
    def action_space(self):
        self._action_space

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)
        
    def transition_prob(self, state_t, state_tp1):
        """
        Returns the probability of transitioning from 
        state_t to state_tp1 in the stochastic process
        """
        raise NotImplementedError
    
    def reset(self):
        pass
    
    def reset_task(self):
        self.simulate(self.T)
        
    def simulate(self, T):
        raise NotImplementedError


class RandomWalk(StochasticProcess):
    def __init__(self,
                 dimensions,
                 step_probs,
                 step_sizes,
                 # dt=1,
                 seed=10,
                 T=100,
                 n_agents=1):
        """
        This will simulate a (discrete) random walk in :dimensions: dimensions.
        The probability of transisitioning to a different state is given by step_probs
        The corresponding states are given in step_sizes
        :param dimensions: the number of dimensions the problem has
        :param step_probs: list containing 2*dimensions + 1
        :param step_sizes: a list containing 2*dimensions+1 elements of size dimensions
                    indicating the step sizes for each move
        # :param dt: the time between successive events
        :param seed: the seed for the internal random number generator
        :param T: the total number of time steps to simulate for
        :param n_agents: the number of agents that will interact with the trajectory
        """
        super().__init__(seed, T)
        assert len(step_probs) == 2*dimensions+1, 'You must supply {} step_probs for dimensions'.format(2*dimensions+1, dimensions)
        assert len(step_sizes) == 2*dimensions+1, 'You must supply {} step_sizes for dimensions'.format(2*dimensions+1, dimensions)
        assert sum([len(step_size) == dimensions for step_size in step_sizes]), 'Each element of step_sizes must be of length {}'.format(dimensions)
        assert sum(step_probs) == 1, 'Step probs must sum to 1.'
        self.step_probs = step_probs
        self.dimensions = dimensions
        self.step_sizes = step_sizes
        # self.dt = dt
        self.x0 = np.zeros(dimensions)
        self.state = self.x0
        self._state_space = dimensions
        self._action_space = 2*dimensions + 1
        self.n_agents = n_agents
        self.new_task()
    
    def simulate(self, rng=None):
        """
        This simulates a trajectory and stores it in the memory
        """
        if rng is None:
            rng = self.rng
        steps_idx = rng.multinomial(1, self.step_probs, self.T).argmax(axis=1)
        steps_taken = np.take(self.step_sizes, steps_idx, axis=0)
        steps_taken = np.vstack([self.x0, steps_taken])
        return steps_taken.cumsum(axis=0)

    def reset_agent_locations(self):
        """
        This will reset the locations of all the agents who are currently interacting with
        the stochastic process
        """
        self.global_time_step = 0
        self.x_agent = np.repeat(self.xT.reshape(1, self.dimensions), self.n_agents, axis=0)

    def reset(self):
        """
        This resets the game
        """
        self.reset_agent_locations()
        return self.x_agent
    
    def new_task(self):
        """
        This will create a new task/stochastic process for the agents to interact with
        """
        self.true_trajectory = self.simulate()
        self.xT = self.true_trajectory[-1]
        return self.reset()

    def step(self, actions, reverse=True):
        """
        This will allow the agents to execute the actions in the stochastic process.
        :param actions: the array with the actions to be executed by each agent
        :param reverse: defines if the step execution is going in reverse or not
        """
        if self.global_time_step == self.T:
            raise TimeoutError('You have already reached the end of the episode. Use reset()')
        steps_taken = np.take(self.step_sizes, actions.ravel(), axis=0)
        step_log_probs = np.log(np.take(self.step_probs, actions.ravel(), axis=0).reshape(self.n_agents, -1))

        reversal_param = -1 if reverse else +1
        self.x_agent = self.x_agent + steps_taken * reversal_param
        self.global_time_step += 1
        if self.global_time_step == self.T:
            # this will count the agents who reached the correct entry in some state
            corrects = (self.x_agent == self.x0).sum(axis=1)
            # print(corrects)
            # the final log prob is just the sum of getting to the correct
            # position in this space.
            step_log_probs = np.log(corrects * 1/self.dimensions + np.finfo('float').tiny)
        return (self.x_agent, step_log_probs.reshape(-1, 1), self.global_time_step == self.T, {})
