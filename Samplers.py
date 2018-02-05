import numpy as np
from utils import SamplingResults

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
        results = SamplingResults('ABCSampler')
        trajectories = []
        all_trajectories = []
        observed_ending_location = stochastic_process.xT
        starting_location = stochastic_process.x0
        for i in range(mc_samples):
            trajectory = stochastic_process.simulate(self.rng)
            final_position = trajectory[-1]
            # only accept this trajectory if it ended close to the final
            if np.sum(np.abs(observed_ending_location - final_position)) < self.tolerance:
                trajectories.append(trajectory)
            all_trajectories.append(trajectory)

        results.all_trajectories(all_trajectories)
        results.trajectories(trajectories)
        results.create_posterior()
        return results


class MCSampler(Sampler):
    def __init__(self, log_prob_tolerance=-10**10, seed=0):
        super().__init__(seed)
        # self.start_state = start_state
        self.log_prob_tolerance = log_prob_tolerance

    def draw_step(self, current_state, time_to_end=None):
        steps_idx = self.rng.multinomial(1, self.step_probs, 1).argmax(axis=1)
        steps_taken = np.take(self.step_sizes, steps_idx, axis=0)
        step_log_probs = np.log(np.take(self.step_probs, steps_idx, axis=0)).sum()
        return steps_idx, steps_taken, step_log_probs

    def solve(self, stochastic_process, mc_samples):
        results = SamplingResults('MCSampler')
        self.step_probs, self.step_sizes = stochastic_process.step_probs, stochastic_process.step_sizes
        trajectories = []
        all_trajectories = []
        observed_ending_location = stochastic_process.xT
        x_0 = stochastic_process.x0

        for i in range(mc_samples):
            x_t = stochastic_process.reset()  # start at the end
            trajectory_i = [observed_ending_location]
            log_path_prob = 0

            # go in reverse time:
            for t in reversed(range(0, stochastic_process.T-1)):
                x_t = trajectory_i[-1]
                # draw a reverse step
                # this is p(w_{t} | w_{t+1})
                step_idx, step, proposal_log_prob = self.draw_step(x_t)
                x_t, path_log_prob, _, _ = stochastic_process.step(step_idx)
                # probability of the path gets updated:
                log_path_prob += path_log_prob
                # take the reverse step:
                trajectory_i.append(x_t)

            if log_path_prob > -np.inf:
                trajectories.append(np.vstack(list(reversed(trajectory_i))))
            all_trajectories.append(np.vstack(list(reversed(trajectory_i))))

        results.all_trajectories(all_trajectories)
        results.trajectories(trajectories)
        results.create_posterior()
        return results


class ISSampler(Sampler):
    def __init__(self, proposal, log_prob_tolerance=-10**4, seed=0):
        super().__init__(seed)
        self.proposal = proposal
        self.log_prob_tolerance = log_prob_tolerance

    def solve(self, stochastic_process, mc_samples):
        results = SamplingResults('ISSampler')
        proposal = self.proposal(stochastic_process.x0, stochastic_process.step_sizes, rng=self.rng)
        trajectories = []
        posterior_particles = []
        posterior_weights = []
        all_trajectories = []
        observed_ending_location = stochastic_process.xT
        x_0 = stochastic_process.x0

        for i in range(mc_samples):
            x_t = stochastic_process.reset()  # start at the end
            trajectory_i = [x_t]
            log_path_prob = 0
            log_proposal_prob = 0
            # go in reverse time:
            for t in reversed(range(0, stochastic_process.T-1)):
                x_t = trajectory_i[-1]
                # draw a reverse step
                # this is p(w_{t} | w_{t+1})
                step_idx, step, log_prob_proposal_step = proposal.draw(x_t, t)
                # print('proposal_log_prob step:',log_prob_proposal_step)
                x_t, path_log_prob, _, _ = stochastic_process.step(step_idx)
                # print('path_log_prob step:',path_log_prob)

                # probability of the path gets updated:
                log_path_prob += path_log_prob
                log_proposal_prob += log_prob_proposal_step
                # take the reverse step:
                trajectory_i.append(x_t)


            likelihood_ratio = log_path_prob - log_proposal_prob

            if log_path_prob > -np.inf:
                # print()
                trajectories.append(np.vstack(list(reversed(trajectory_i))))
                posterior_particles.append(trajectories[-1][0])
                posterior_weights.append(np.exp(likelihood_ratio))
            all_trajectories.append(np.vstack(list(reversed(trajectory_i))))

        results.all_trajectories(all_trajectories)
        results.trajectories(trajectories)
        results.posterior_weights(posterior_weights)
        results.posterior(posterior_particles)
        return results
