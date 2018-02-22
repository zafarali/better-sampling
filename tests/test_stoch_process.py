import numpy as np
from rvi_sampling.StochasticProcess import RandomWalk

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

# def