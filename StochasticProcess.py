import torch
import torch.nn.functional as F
from Containers import Trajectory

USE_CUDA = torch.cuda.is_available()

class StochasticProcess(object):
    def __init__(self, seed=10):
        torch.manual_seed(seed)

    def transition_prob(self, st, stp1):
        raise NotImplementedError('You need to implement this')

    def simulate(self, T):
        raise NotImplementedError('You need to implement this')

class BiasedRandomWalk1D(StochasticProcess):
    def __init__(self,
                 step_probs=[1/3, 1/3, 1/3],
                 step_size=1,
                 dt=1,
                 seed=10):
        """
        Creates a biased random walk in 1D
        :param step_probs:
        :param step_size: size of each step
        :param dt: the time between steps
        :param seed: the random seed to use
        """
        super().__init__(seed)
        self.step_probs = torch.Tensor(step_probs)
        self.step_size = step_size
        self.dt = dt
        self.seed = seed
        self.x0 = 0

    def draw_step(self):
        return self.step_probs.multinomial(1)[0]-1

    def transition_prob(self, st, stp1):
        # get the difference between two consecutive steps
        # and find out the probability of such a step
        step = int(((stp1 - st) / self.step_size)+1)
        return self.step_probs[step]

    def simulate(self, T):
        """
        Simulates a random walk trajectory
        """
        trajectory = Trajectory()

        x = self.x0
        trajectory.new_step(0, x)

        for t in torch.arange(self.dt, T, self.dt):
            step = self.draw_step()
            x += step
            trajectory.new_step(t, x)

        self.last_trajectory = trajectory
        return trajectory

class RandomWalk1D(BiasedRandomWalk1D):
    def __init__(self,
                 allow_no_step=False,
                 step_size=1,
                 dt=1,
                 seed=10):
        """
        Creates an unbiased random walk in 1D
        :param allow_no_step: allows the walk to take no step
        :param step_size: size of each step
        :param dt: the time between steps
        :param seed: the random seed to use
        """
        if allow_no_step:
            super().__init__(step_size=step_size,
                             dt=dt,
                             seed=seed)
        else:
            super().__init__(step_probs=[1/2, 0, 1/2],
                             dt=dt,
                             step_size=step_size,
                             seed=seed)

class RandomWalknD(StochasticProcess):
    def __init__(self, n=2, allow_no_step=False):
        """
        A random walk in n dimensions
        """

        # a random walk in n dimensions
        # can just be viewed as n independent random walks
        self.walks = [ RandomWalk1D(allow_no_step) for _ in range(n)]

    def draw_step(self):
        return [ walk.draw_step() for walk in self.walks ]

    def transition_prob(self, transitions):
        """
        :param transitions: a tuple (st, stp1) of transitions for each walk
        """
        probs = []
        for walk, transition in zip(self.walks, transitions):
            probs.append(walk.transition_prob(*transition))
        return probs

    def simulate(self, T):
        xs = []
        ts = None

        for walk in self.walks:
            trajectory = walk.simulate(T)
            xs.append(trajectory.x)
            ts = trajectory.t
        
        return Trajectory().extend(ts, zip(*xs))