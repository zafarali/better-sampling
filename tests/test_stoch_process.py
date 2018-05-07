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
        i+= 1
    assert i == 3

def test_stochastic_process_step_calls_actual_steps_multiple_agents():
    rw = RandomWalk(DIMENSIONS, STEP_PROBS, POSSIBLE_STEPS, T=T, n_agents=2)

    agent_location = rw.reset()
    assert agent_location.shape == (2, 1)
    new_location, reward, _, _ = rw.step(np.array([[0, 0]])) # take two left steps.
    assert np.all(reward == np.log(0.5))
    assert np.all(new_location - agent_location == -1)

    agent_location = new_location
    new_location, reward, _, _ = rw.step(np.array([[0, 1]]))  # take one left step and one right step
    assert np.all(reward == np.log(0.5))
    assert tuple((new_location - agent_location).reshape(-1).tolist()) == (-1, 1)


def test_stochastic_process_reverse_steps_and_rewards():
    rw = RandomWalk(DIMENSIONS, STEP_PROBS, POSSIBLE_STEPS, T=3, prior_distribution=DiscreteUniform(1, -2, 4), n_agents=2)

    rw.x_agent = np.array([[-1], [-1]])
    agent_location = rw.x_agent
    assert agent_location.shape == (2, 1)
    new_location, reward, done, _ = rw.step(np.array([[0, 0]]), reverse=True) # take two left steps.
    assert np.allclose(reward, np.log(0.5))
    assert np.all(new_location - agent_location == +1)
    assert not done
    print(new_location)
    agent_location = new_location
    new_location, reward, done, _ = rw.step(np.array([[1, 0]]), reverse=True)  # take one left step and one right step
    print(new_location)
    assert done # three steps is 2 transitions.
    assert np.allclose(reward, np.log(0.5) + np.log(1/4))
    assert tuple((new_location - agent_location).reshape(-1).tolist()) == (-1, +1)

    # now a walk that goes out of the prior.
    rw.reset()
    rw.x_agent = np.array([[-1], [-1]])
    agent_location = rw.x_agent
    assert agent_location.shape == (2, 1)
    new_location, reward, done, _ = rw.step(np.array([[1, 1]]), reverse=True) # take two left steps.
    assert np.allclose(reward, np.log(0.5))
    assert np.all(new_location - agent_location == -1)
    assert not done
    print(new_location)
    agent_location = new_location
    new_location, reward, done, _ = rw.step(np.array([[1, 1]]), reverse=True)  # take one left step and one right step
    print(new_location)
    assert done # three steps is 2 transitions.
    assert np.allclose(reward, np.log(0))
    assert tuple((new_location).reshape(-1).tolist()) == (-3, -3)

def test_2d_process():
    rw = RandomWalk(2, [0.4, 0.6], [[-1, 1], [0, 1]], T=4, prior_distribution=DiscreteUniform(2, 0, 2))
    x = rw.reset()
    assert x.shape == (1, 2)

    pytrw = PyTorchWrap(rw)

    x = pytrw.reset()
    assert isinstance(x, Variable)
    assert tuple(x.size()) == (1, 2)