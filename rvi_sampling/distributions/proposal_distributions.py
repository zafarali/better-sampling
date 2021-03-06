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


    # TODO(zaf): IDEA!
    # IDEA for speed up: Let draw_legacy return only the sampling probs for each row
    # We then use the efficient multinomial sampling to get the action to be taken
    # and the corresponding log probability in this function to return that.
    def draw(self, w_batch, time_left, sampling_probs_only=False):
        """
        :param w_batch: A np.ndarray containing the batch of positions of shape
            (B, N) where N is the dimensionality of the stochastic process
        and B is the batch size.
        :param time_left: An int representing the time left until the process
            is over.
        :param sampling_probs_only: A bool indicating if the return will be
            probabilities of shape (B, A) where A is the number of possible
            actions. If False it returns a tuple containing the sampled action
            index and the log_prob of taking that action.
        """
        # Batchify the drawing function.
        return_value = np.apply_along_axis(
            func1d=self.draw_legacy,
            axis=1,
            arr=w_batch,
            time_left=time_left,
            sampling_probs_only=sampling_probs_only)

        if sampling_probs_only:
            return return_value
        else:
            return (return_value[:, 0].astype(int),
                    return_value[:, 1].astype(np.float))

            # TODO(zaf): Improvement idea goes something like this:
            # samples = sampling_utils.vectorized_multinomial(
            #     sampling_probs,
            #     np.arange(sampling_probs.shape[1])
            # ).astype(int)
            # print(samples)
            # log_probs = np.log(sampling_probs[:, samples])[:, 0]
            # print(log_probs)
            # return samples, log_probs

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
        w = np.array([w])
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
                return np.concatenate((
                    np.array([index]),
                    np.array([np.log(1/2.)])
                    ))
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
        choice_prob = probs[choice_index]

        return np.concatenate((choice_index, np.log(choice_prob)))

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
        return np.array([index]), np.log(1 / 2)

class FunnelProposal(ProposalDistribution):
    _soft = True
    def __init__(self, push_toward=[-5, 5], step_sizes=None, seed=0, rng=None):
        super().__init__(seed)
        self.push_toward = push_toward
        if rng:
            self.rng = rng


    def draw(self, w_batch, time_left, sampling_probs_only=False):
        """
        :param w_batch: A np.ndarray containing the batch of positions of shape
            (B, N) where N is the dimensionality of the stochastic process
        and B is the batch size.
        :param time_left: An int representing the time left until the process
            is over.
        :param sampling_probs_only: A bool indicating if the return will be
            probabilities of shape (B, A) where A is the number of possible
            actions. If False it returns a tuple containing the sampled action
            index and the log_prob of taking that action.
        """
        # Batchify the drawing function.
        return_value = np.apply_along_axis(
            func1d=self.draw_legacy,
            axis=1,
            arr=w_batch,
            time_left=time_left,
            sampling_probs_only=sampling_probs_only)

        if sampling_probs_only:
            return return_value
        else:
            return (return_value[:, 0].astype(int),
                    return_value[:, 1].astype(np.float))

    def draw_legacy(self, w, time_left, sampling_probs_only=False):

        STEP_SIZE = 1

        # assuming symmetric window
        steps_left = STEP_SIZE * time_left

        if np.abs(w - self.push_toward[0]) > steps_left \
                or np.abs(w-self.push_toward[1]) > steps_left:
            # we are not near the window boundary
            # we now check how far, if based on the time available
            # we cannot ever move into the window, we push hard toward the window.
            if np.sign(w) == -1 and self.push_toward[0] - (w -1) > steps_left-1:
                # too much in the negative direction
                if sampling_probs_only:
                    return np.array([0, 1])
                return np.array([1]), np.log(1 - np.finfo(float).eps)

            elif np.sign(w) == +1 and (w+1) - self.push_toward[1] > steps_left-1:
                # we are in the positive area, if we took a step
                # in the positive direction in the next time step
                # would we still have enough steps to be in the window?
                if sampling_probs_only:
                    return np.array([1, 0])
                return np.array([0]), np.log(1-np.finfo(float).eps)

        probs = np.array([1 / 2., 1 / 2.])

        if sampling_probs_only:
            return probs

        choice_index = np.array([self.rng.multinomial(1, probs, 1).argmax()])
        choice_prob = probs[choice_index]
        return np.concatenate((choice_index, np.log(choice_prob)))

