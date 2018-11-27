import logging
from scipy.spatial.distance import cosine
import numpy as np
from rvi_sampling.distributions import sampling_utils


class ProposalDistribution(object):
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def draw(self, x: object, time_left: object) -> object:
        raise NotImplementedError

    def set_rng(self, rng):
        self.rng = rng

class MinimalProposal(ProposalDistribution):
    def __init__(self, push_toward, step_sizes, bias=0, seed=0, rng=None):
        logging.warning('This proposal is not tested.')
        self.push_toward = push_toward
        self.bias = bias
        self.step_sizes = step_sizes
        super().__init__(seed)
        if rng:
            self.rng = rng

    def draw(self, x, time_left, sampling_probs_only=False):
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

        if sampling_probs_only:
            return sampling_probs

        chosen_step_idx = self.rng.multinomial(1, sampling_probs, 1).argmax()
        chosen_step = self.step_sizes[chosen_step_idx]
        step_prob = sampling_probs[chosen_step_idx]

        return chosen_step_idx, chosen_step, np.log(step_prob)

class SimonsProposal(ProposalDistribution):
    _soft = False
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


class SimonsSoftProposal(SimonsProposal):
    _soft = True
    def __init__(self, push_toward=[-5, 5], step_sizes=None, seed=0, rng=None, softness_coeff=1.0):
        assert len(push_toward) == 2, 'push toward expects a window range'
        super().__init__(step_sizes=step_sizes, seed=seed, rng=rng)
        self.push_toward = np.array(push_toward)
        self.softness_coeff = softness_coeff

    # TODO(zaf): Test this.
    def draw(self, w, time_left, sampling_probs_only=False):
        assert w.shape[1] == 1
        w = np.squeeze(w, -1)  # assumption here is that it is 1D
        n_agents = w.shape[0]
        n_actions = 2
        log_probs = np.zeros((n_agents,))
        action_index = np.zeros((n_agents,))
        sampling_probs = np.ones((n_agents, n_actions))
        step_size = 1

        push_toward, softness_coeff = self.push_toward, self.softness_coeff

        def _determine_bias(wi, time_left):
            if wi < push_toward[0]:
                return (push_toward[0] - wi) * softness_coeff / time_left
            else:  # wi > push_toward[1]:
                return (push_toward[1] - wi) * softness_coeff / time_left

        determine_bias = np.vectorize(_determine_bias)

        bias_vector = determine_bias(w, time_left)
        # These indices are never going to make it to the end. Just be random.
        never_going_to_make_it = np.where(np.abs(bias_vector) > step_size)
        sampling_probs[never_going_to_make_it] /= 2
        log_probs[never_going_to_make_it] = np.log(1.0 / 2.0)
        action_index[never_going_to_make_it] = self.rng.randint(
            0, n_actions, n_agents)

        # Potentially successful
        might_make_it = np.where(np.abs(bias_vector) <= step_size)
        bias_vector_might_make_it = bias_vector[might_make_it]
        p = 1 - np.abs(bias_vector_might_make_it) / step_size
        bias_prob_term = np.zeros((n_agents, 2))[might_make_it]

        bias_prob_term[np.where(bias_vector_might_make_it > 0), 1] = 1
        bias_prob_term[np.where(bias_vector_might_make_it < 0), 0] = 1
        # print('bias_prob_term:', bias_prob_term)
        # print(bias_prob_term.shape)
        if bias_prob_term.shape[0] != 0:
            probs = (
                np.array([p / 2., p / 2.]).reshape(*bias_prob_term.shape)
                + (1 - p) * bias_prob_term
            )
            # print('probs', probs)
            sampling_probs[might_make_it, :] = probs
            # print('sampling probs',sampling_probs)

            choice_index = sampling_utils.vectorized_multinomial(
                probs, np.arange(n_actions))
            # print('choice_index:', choice_index)
            # print('probs:', probs)
            # print('n_actions:', np.arange(n_actions))

            choice_prob = probs[:, choice_index]

            log_probs[might_make_it] = np.log(choice_prob).sum(1)
            action_index[might_make_it] = choice_index

        if sampling_probs_only:
            return sampling_probs
        else:
            # none is here for backward compatability.
            return action_index.astype(int), None, log_probs



    def draw_legacy(self, w, time_left, sampling_probs_only=False):
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
        #
        if np.abs(w) > np.abs(self.push_toward[0]):
            # bias = (sign * np.abs(self.push_toward[0]) - w) * 1. / time_left
            if w < self.push_toward[0]: # x < -c
                bias = (self.push_toward[0] - w) * self.softness_coeff / time_left
                # distance to acceptable position is c - w. Average steps needed is c-w/T
            elif w > self.push_toward[1]: # x > c
                 bias = (self.push_toward[1] - w) * self.softness_coeff / time_left
            else:
                # this conditional will never get executed for now.
                bias = (sign*np.abs(self.push_toward[1]) - w) * self.softness_coeff / time_left


        bias = bias[0][0]
        # print('time: {}, w: {}, bias: {} '.format(time_left, w, bias))

        # if the bias needed is more than the actual steps per time that you can take
        # we're basically never going to make it so we skip this
        if np.abs(bias) > STEP_SIZE:
            # print("will fail, might as well stop now.")
            random_step = np.array([2*self.rng.randint(0,2)-1])
            if random_step == -1:
                index = 0
            else:
                index = 1

            if sampling_probs_only:
                return np.array([1., 1.])/2
            else:
                return np.array([index]), random_step, np.log(1/2)
            
        # with probability p, pick uniform sampling; with probability 1-p, pick a step in the bias direction.
        # expected bias is (1-p) * step_size
        p = 1 - np.abs(bias) / STEP_SIZE
        bias_prob_term = np.array([0, 0])
        if bias > 0:
            bias_prob_term[1] = 1
        if bias < 0:
            bias_prob_term[0] = 1
        choices = np.array([[-STEP_SIZE], [STEP_SIZE]])

        probs = np.array([p / 2., p / 2.]) + (1 - p) * bias_prob_term
        # probs = np.exp(probs)
        # probs /= probs.sum()
        if sampling_probs_only:
            return probs

        choice_index = np.array([self.rng.multinomial(1, probs, 1).argmax()])
        choice_step = choices[choice_index]
        choice_prob = probs[choice_index]

        return choice_index, choice_step, np.log(choice_prob)[0]


class RandomProposal(ProposalDistribution):
    _soft = False
    """
    A random proposal for the 1D random walk.
    Only works for two possible steps: -1 and +1
    """
    def __init__(self, push_toward=[0], step_sizes=None, seed=0, rng=None):
        assert len(push_toward) == 1, 'This proposal only works in 1D'
        super().__init__(seed)
        if rng:
            self.rng = rng


    def draw(self, x, time_left, sampling_probs_only=False):
        if sampling_probs_only:
            return np.array([0.5, 0.5])

        random_step = np.array([2 * self.rng.randint(0, 2) - 1])
        if random_step == -1:
            index = 0
        else:
            index = 1
        return np.array([index]), random_step, np.log(1 / 2)

class FunnelProposal(ProposalDistribution):
    _soft = True
    def __init__(self, push_toward=[-5, 5], step_sizes=None, seed=0, rng=None):
        super().__init__(seed)
        self.push_toward = push_toward
        if rng:
            self.rng = rng

    def draw(self, w, time_left, sampling_probs_only=False):

        STEP_SIZE = 1
        # print(w)
        w = w[0][0]
        # print(w)

        # assuming symmetric window
        steps_left = STEP_SIZE * (time_left)

        if np.abs(w - self.push_toward[0]) > steps_left \
                or np.abs(w-self.push_toward[1]) > steps_left:
            # we are not near the window boundary
            # we now check how far, if based on the time available
            # we cannot ever move into the window, we push hard toward the window.
            if np.sign(w) == -1 and self.push_toward[0] - (w -1) > steps_left-1:
                # too much in the negative direction
                if sampling_probs_only:
                    return np.array([0, 1])
                return np.array([1]), +1, np.log(1 - np.finfo(float).eps)

            elif np.sign(w) == +1 and (w+1) - self.push_toward[1] > steps_left-1:
                # we are in the positive area, if we took a step
                # in the positive direction in the next time step
                # would we still have enough steps to be in the window?
                if sampling_probs_only:
                    return np.array([1, 0])
                return np.array([0]), -1, np.log(1-np.finfo(float).eps)

        choices = np.array([[-STEP_SIZE], [STEP_SIZE]])

        probs = np.array([1 / 2., 1 / 2.])

        if sampling_probs_only:
            return probs

        choice_index = np.array([self.rng.multinomial(1, probs, 1).argmax()])
        choice_step = choices[choice_index]
        choice_prob = probs[choice_index]

        return choice_index, choice_step, np.log(choice_prob)[0]