import gym
import numpy as np
import torch
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


class PyTorchWrap(object):
    _pytorch = True
    _vectorized = True
    def __init__(self, stochastic_process, use_cuda=False):
        self.stochastic_process = stochastic_process
        # self.step_probs = stochastic_process.step_probs or stochastic_process.transition_prob
        self.dimensions = stochastic_process.dimensions
        self.step_sizes = stochastic_process.step_sizes
        self.x0 = stochastic_process.x0
        self.state_space = stochastic_process.state_space
        self.action_space = stochastic_process.action_space
        self.use_cuda = use_cuda
        self.T = stochastic_process.T
        self.n_agents = stochastic_process.n_agents
        self.true_trajectory = self.stochastic_process.true_trajectory
        self._training = True

    @property
    def xT(self):
        return self.stochastic_process.xT

    def train_mode(self, mode):
        self._training = mode
    @property
    def transitions_left(self):
        return self.stochastic_process.transitions_left

    def variable_wrap(self, tensor):
        return tensor.float()

    def simulate(self, rng=None):
        return self.stochastic_process.simulate(rng=rng)

    def reset(self):
        return self.variable_wrap(torch.from_numpy(self.stochastic_process.reset()))

    def prior_pdf(self, x_t):
        x_t = x_t.data.numpy()
        return self.stochastic_process.prior.pdf(x_t)

    def new_task(self):
        delayed_to_return = self.stochastic_process.new_task()
        self.true_trajectory = self.stochastic_process.true_trajectory
        return delayed_to_return

    def step(self, actions, reverse=True):
        actions = actions.numpy()
        position, log_probs, done, info = self.stochastic_process.step(actions, reverse=reverse)
        position = self.variable_wrap(torch.from_numpy(position))
        log_probs = torch.from_numpy(log_probs).float().view(self.n_agents, 1)
        done = torch.IntTensor([[done] * self.n_agents])
        return (position, log_probs, done, info)


class StochasticProcessOverError(TimeoutError):
    def __init__(self, text='Stochastic Process is complete, you must reset()', *args, **kwargs):
        super().__init__(text, *args, **kwargs)
