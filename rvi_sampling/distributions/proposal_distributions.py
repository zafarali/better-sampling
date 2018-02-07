import random
from scipy.spatial.distance import cosine
import numpy as np

class ProposalDistribution(object):
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def draw(self, x: object, time_left: object) -> object:
        raise NotImplementedError

class MinimalProposal(ProposalDistribution):
    def __init__(self, push_toward, step_sizes, bias=0, seed=0, rng=None):
        self.push_toward = push_toward
        self.bias = bias
        self.step_sizes = step_sizes
        super().__init__(seed)
        if rng:
            self.rng = rng

    def draw(self, x, time_left):
        """
        A slight pushing distribution

        :param x: the current position
        :param time_left: the number of time steps left until we reach the start
        :return:
        """
        target = self.push_toward
        current = x
        direction = target - current

        unnormalized_bias_probs = -np.array([cosine(step + np.finfo('float').tiny, direction) for step in self.step_sizes])
        bias_probs = np.exp(unnormalized_bias_probs)
        bias_probs /= bias_probs.sum()
        unbias_probs = np.ones(len(self.step_sizes)) * 1/len(self.step_sizes)

        steps_needed = np.abs(direction).sum()
        # estimates the amount of bias to be implemented
        # based on how many steps needed to go and how much time you have left.
        step_ratio = time_left/steps_needed
        p = np.min([step_ratio, 0.999999])

        sampling_probs = unbias_probs * p + (1-p)*bias_probs
        sampling_probs /= sampling_probs.sum()

        chosen_step_idx = self.rng.multinomial(1, sampling_probs, 1).argmax()
        chosen_step = self.step_sizes[chosen_step_idx]
        step_prob = sampling_probs[chosen_step_idx]

        return chosen_step_idx, chosen_step, np.log(step_prob)

class SimonsProposal(ProposalDistribution):
    """
    A proposal for the 1D random walk.
    Only works for two possible steps: -1 and +1
    """
    def __init__(self, push_toward=[0], step_sizes=None, seed=0, rng=None):
        assert len(push_toward) == 1, 'This proposal only works in 1D'
        super().__init__(seed)
        if rng:
            self.rng = rng
        self.push_toward = np.array(push_toward)
        if step_sizes is not None:
            # used to check if the steps in the RW process
            # are compatible with this proposal
            assert len(step_sizes) == 2 and step_sizes[0][0] == -1 and step_sizes[1][0] == +1, \
                  'RW step distribution is malformed.'


    def draw(self, w, time_left):
        """
        :param w: the current position
        :param time_left: the number of time steps until we reach the start
        :return:
        """
        # w is kept for historical reasons, it is actually x
        #     print('pos:',w)
        #     print('toward:',push_toward)

        # We want to push slightly, such that the average step is d/T, where d is distance to
        # the acceptable position, and T is the number of generations left.
        bias = np.array([[0]])
        STEP_SIZE = 1

        sign = w / np.abs(w)

        if np.abs(w) > np.abs(self.push_toward):
            bias = (sign * self.push_toward - w) * 1. / time_left

        bias = bias[0][0]
        # with probability p, pick uniform sampling; with probability 1-p, pick a step in the bias direction.
        # expected bias is (1-p) * step_size
        if np.abs(bias) > STEP_SIZE:
            # print("will fail, might as well stop now.")
            random_step = np.array([2*self.rng.randint(0,2)-1])
            if random_step == -1:
                index = 0
            else:
                index = 1
            return np.array([index]), random_step, np.log(1/2)

        p = 1 - np.abs(bias) / STEP_SIZE
        bias_prob_term = np.array([0, 0])
        if bias > 0:
            bias_prob_term[1] = 1
        if bias < 0:
            bias_prob_term[0] = 1
        choices = np.array([[-STEP_SIZE], [STEP_SIZE]])

        probs = np.array([p / 2., p / 2.]) + (1 - p) * bias_prob_term
        # print(probs)
        choice_index = np.array([self.rng.multinomial(1, probs, 1).argmax()])
        choice_step = choices[choice_index]
        choice_prob = probs[choice_index]

        return choice_index, choice_step, np.log(choice_prob)[0]

#