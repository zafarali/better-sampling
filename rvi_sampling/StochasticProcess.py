import gym
import numpy as np
import torch
from torch.autograd import Variable
from .distributions.prior_distributions import DiscreteUniform
from functools import reduce

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
                 n_agents=1,
                 prior_distribution=DiscreteUniform(1, 0, 2)):
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
        # remove these requirements for now since we can specify a more diverse set of possible steps.
        # assert len(step_probs) == 2*dimensions+1, 'You must supply {} step_probs for dimensions'.format(2*dimensions+1, dimensions)
        # assert len(step_sizes) == 2*dimensions+1, 'You must supply {} step_sizes for dimensions'.format(2*dimensions+1, dimensions)
        assert sum([len(step_size) == dimensions for step_size in step_sizes]), 'Each element of step_sizes must be of length {}'.format(dimensions)
        assert sum(step_probs) == 1, 'Step probs must sum to 1.'
        self.step_probs = step_probs
        self.dimensions = dimensions
        self.step_sizes = step_sizes
        # self.dt = dt
        # self.state = self.x0
        self._state_space = dimensions
        self._action_space = 2*dimensions + 1
        self.n_agents = n_agents
        self.prior = prior_distribution
        self.new_task()

    
    def simulate(self, rng=None):
        """
        This simulates a trajectory and stores it in the memory
        """
        if rng is None:
            rng = self.rng
        x0 = self.prior.rvs()
        # T steps is T-1 transitions
        steps_idx = rng.multinomial(1, self.step_probs, self.T-1).argmax(axis=1)
        steps_taken = np.take(self.step_sizes, steps_idx, axis=0)
        steps_taken = np.vstack([x0, steps_taken])
        return steps_taken.cumsum(axis=0)

    def reset_agent_locations(self):
        """
        This will reset the locations of all the agents who are currently interacting with
        the stochastic process
        """
        self.transitions_left = self.T-1
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
        self.x0 = self.true_trajectory[0]
        self.xT = self.true_trajectory[-1]
        return self.reset()

    def step(self, actions, reverse=False):
        """
        This will allow the agents to execute the actions in the stochastic process.
        :param actions: the array with the actions to be executed by each agent
        :param reverse: defines if the step execution is going in reverse or not
        """
        if self.transitions_left == 0:
            #TODO return a custom error here to not conflate it with builtin types.
            raise TimeoutError('You have already reached the end of the episode. Use reset()')

        steps_taken = np.take(self.step_sizes, actions.ravel(), axis=0)
        step_log_probs = np.log(np.take(self.step_probs, actions.ravel(), axis=0).reshape(self.n_agents, -1))

        reversal_param = -1 if reverse else +1
        self.x_agent = self.x_agent + (steps_taken * reversal_param)
        self.transitions_left -= 1
        if self.transitions_left == 0:
            # if there are no more transitions left, also add the "final reward"
            # to the step_log_probs
            step_log_probs += np.log(self.prior.pdf(self.x_agent))

        return (self.x_agent, step_log_probs.reshape(-1, 1), self.transitions_left == 0, {})


class PyTorchWrap(object):
    _pytorch = True
    _vectorized = True
    def __init__(self, stochastic_process, use_cuda=False):
        self.stochastic_process = stochastic_process
        self.step_probs = stochastic_process.step_probs
        self.dimensions = stochastic_process.dimensions
        self.step_sizes = stochastic_process.step_sizes
        self.x0 = stochastic_process.x0
        self.state_space = stochastic_process.state_space
        self.action_space = stochastic_process.action_space
        self.use_cuda = use_cuda
        self.T = stochastic_process.T
        self.n_agents = stochastic_process.n_agents
        self.xT = stochastic_process.xT
        self.true_trajectory = self.stochastic_process.true_trajectory
        self._training = True

    def train_mode(self, mode):
        self._training = mode
    @property
    def transitions_left(self):
        return self.stochastic_process.transitions_left

    def variable_wrap(self, tensor):
        if not isinstance(tensor, Variable):
            if self._training:
                tensor = Variable(tensor)
            else:
                tensor = Variable(tensor, volatile=True)
        if self.use_cuda:
            tensor = tensor.cuda()

        return tensor.float()

    def simulate(self, rng=None):
        return self.stochastic_process.simulate(rng=rng)

    def reset(self):
        return self.variable_wrap(torch.from_numpy(self.stochastic_process.reset()))

    def prior_pdf(self, x_t):
        x_t = x_t.cpu().data.numpy()
        return self.stochastic_process.prior.pdf(x_t)

    def new_task(self):
        delayed_to_return = self.stochastic_process.new_task()
        self.xT = self.stochastic_process.xT
        self.true_trajectory = self.stochastic_process.true_trajectory
        return delayed_to_return

    def step(self, actions, reverse=True):
        if isinstance(actions, Variable):
            actions = actions.data
        actions = actions.cpu()
        actions = actions.numpy()
        position, log_probs, done, info = self.stochastic_process.step(actions, reverse=reverse)
        position = self.variable_wrap(torch.from_numpy(position))
        log_probs = torch.from_numpy(log_probs).float().view(self.n_agents, 1)
        done = torch.IntTensor([[done] * self.n_agents])
        return (position, log_probs, done, info)
