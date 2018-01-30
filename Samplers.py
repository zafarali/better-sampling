import numpy as np

class Sampler(object):
    """
    Abstract class that sample from Posteriors 
    """
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
    def solve(self, stochastic_process, mc_samples):
        raise NotImplementedError


class ABCSampler(Sampler):
    """
    Approximate bayesian computation sampling from the posterior.
    It uses the .simulate() method of the StochasticProcess object
    to get trajectories. It them checks to see if the end of the trajectory
    is within a tolerance factor of the observed ending point. If yes,
    it stores this trajectory to be returned later.
    """
    def __init__(self, tolerance, seed=0):
        super().__init__(seed)
        self.tolerance = tolerance

    def solve(self, stochastic_process, mc_samples):
        trajectories = []
        observed_ending_location = stochastic_process.xT
        starting_location = stochastic_process.x0
        for i in range(mc_samples):
            trajectory = stochastic_process.simulate(self.rng)
            final_position = trajectory[-1]
            # only accept this trajectory if it ended close to the final
            if np.sum(np.abs(observed_ending_location - final_position)) < self.tolerance:
                trajectories.append(trajectory)
        return trajectories


class MCSampler(Sampler):
    def __init__(self, step_probs, step_sizes, log_prob_tolerance=-10**10, seed=0):
        super().__init__(seed)
        # self.start_state = start_state
        self.step_sizes = step_sizes
        self.step_probs = step_probs
        self.log_prob_tolerance = log_prob_tolerance

    def draw_step(self, current_state, time_to_end=None):
        steps_idx = self.rng.multinomial(1, self.step_probs, 1).argmax(axis=1)
        steps_taken = np.take(self.step_sizes, steps_idx, axis=0)
        step_log_probs = np.log(np.take(self.step_probs, steps_idx, axis=0)).sum()
        return steps_idx, steps_taken, step_log_probs

    def solve(self, stochastic_process, mc_samples):
        trajectories = []
        observed_ending_location = stochastic_process.xT
        x_0 = stochastic_process.x0

        for i in range(mc_samples):
            x_t = stochastic_process.reset()  # start at the end
            trajectory_i = [observed_ending_location]
            log_path_prob = 0

            # go in reverse time:
            for t in reversed(range(0, stochastic_process.T)):
                x_t = trajectory_i[-1]
                # draw a reverse step
                # this is p(w_{t} | w_{t+1})
                step_idx, step, proposal_log_prob = self.draw_step(x_t)
                x_t, path_log_prob, _, _ = stochastic_process.step(step_idx)
                # probability of the path gets updated:
                log_path_prob += path_log_prob
                # take the reverse step:
                trajectory_i.append(x_t)

            # check if the trajectory ends at the right place
            # if yes, the path prob is +=1

            if np.all(x_t == x_0):
                log_path_prob += np.log(1)
            else:
                log_path_prob += np.log(np.finfo('float').tiny)
            if log_path_prob > self.log_prob_tolerance:
                # print()
                trajectories.append(np.vstack(list(reversed(trajectory_i))))

        return trajectories


class ISSampler(MCSampler):
    pass

class RVISampler(Sampler):
    """
    Reinforced Variational Inference Sampler
    """
    def __init__(self, policy, policy_optimizer, baseline=None,  seed=0):
        super().__init__(seed)
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.baseline = baseline

    def solve(self, stochastic_process, mc_samples):
        pass