import numpy as np

class ProposalDistribution(object):
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def draw(self, x, time_left):
        raise NotImplementedError

class MinimalProposal(ProposalDistribution):
    def __init__(self, push_toward, step_sizes, bias=None, seed=0):
        self.push_toward = push_toward
        self.bias = np.zeros_like(push_toward) if not bias else bias
        self.step_sizes = step_sizes

    def draw(self, x, time_left):
        """
        A slight pushing distribution

        :param x: the current position
        :param time_left: the number of time steps left until we reach the start
        :return:
        """
        sign = x/np.abs(x)

        if np.abs(x) > np.abs(self.push_toward):
            bias = (sign * self.push_toward - x) * 1 / time_left
        else:
            bias = self.bias

        if np.abs(bias) > self.step_sizes

def draw_from_proposal_minimal(w, time_left, bias=0, push_toward=0):

    sign = w / np.abs(w)
    if np.abs(w) > np.abs(push_toward):
        bias = (sign * push_toward - w) * 1. / time_left


        # with probability p, pick uniform sampling; with probability 1-p, pick a step in the bias direction.
    # expected bias is (1-p) * step_size
    step_size = .5
    if np.abs(bias) > step_size:
        # print("will fail, might as well stop now.")
        return 0, 1
    p = 1 - np.abs(bias) / step_size
    bias_prob_term = np.array([0, 0, 0])
    if bias > 0:
        bias_prob_term[2] = 1
    if bias < 0:
        bias_prob_term[0] = 1
    choices = [-step_size, 0, step_size]
    probs = np.array([p / 3., p / 3., p / 3.]) + (1 - p) * bias_prob_term
    choice = np.random.choice(choices, p=probs)
    prob = probs[choices.index(choice)]
    return choice, prob