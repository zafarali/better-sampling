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



#
# Simons original sampling code
# def draw_from_proposal_minimal(w, time_left, bias=0, push_toward=0):
#
#     sign = w / np.abs(w)
#     if np.abs(w) > np.abs(push_toward):
#         bias = (sign * push_toward - w) * 1. / time_left
#
#
#         # with probability p, pick uniform sampling; with probability 1-p, pick a step in the bias direction.
#     # expected bias is (1-p) * step_size
#     step_size = .5
#     if np.abs(bias) > step_size:
#         # print("will fail, might as well stop now.")
#         return 0, 1
#     p = 1 - np.abs(bias) / step_size
#     bias_prob_term = np.array([0, 0, 0])
#     if bias > 0:
#         bias_prob_term[2] = 1
#     if bias < 0:
#         bias_prob_term[0] = 1
#     choices = [-step_size, 0, step_size]
#     probs = np.array([p / 3., p / 3., p / 3.]) + (1 - p) * bias_prob_term
#     choice = np.random.choice(choices, p=probs)
#     prob = probs[choices.index(choice)]
#     return choice, prob