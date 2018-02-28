import numpy as np
from torch.autograd import Variable
from rvi_sampling.stochastic_processes.random_walk import RandomWalk, RWParameters, DiscreteUniform
from rvi_sampling.stochastic_processes.base import PyTorchWrap
POSSIBLE_STEPS = [[-1], [+1]]
STEP_PROBS = [1/2, 1/2]
DIMENSIONS = 1
T = 4

rw = RandomWalk(DIMENSIONS, STEP_PROBS, POSSIBLE_STEPS, T=T)


def test_stochastic_process_length():
    rw.reset()
    assert rw.true_trajectory.shape == (T, DIMENSIONS), 'Length of stochastic process should match T and DIMENSIONS'

def test_stochastic_process_step_calls():
    rw.reset()

    i = 0 # counts the number of `step()` calls or transitions in the VIMDP
    done = False
    while not done:
        _, _, done, _  =rw.step(np.array([[0]]))
        print(done)
        i+= 1
    assert i == 3

def test_2d_process():
    rw = RandomWalk(2, [0.4, 0.6], [[-1, 1], [0, 1]], T=4, prior_distribution=DiscreteUniform(2, 0, 2))
    x = rw.reset()
    assert x.shape == (1, 2)

    pytrw = PyTorchWrap(rw)

    x = pytrw.reset()
    assert isinstance(x, Variable)
    assert tuple(x.size()) == (1, 2)