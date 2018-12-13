import numpy as np
from ..results import SamplingResults, ImportanceSamplingResults
from ..utils import diagnostics

class Sampler(object):
    """
    Abstract class that sample from Posteriors 
    """
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.diagnostic = None
    def solve(self, stochastic_process, mc_samples):
        raise NotImplementedError

    def run_diagnostic(self, results, other_information=None, verbose=False):
        if self.diagnostic is not None:
            result = self.diagnostic(results, other_information)
            if result == diagnostics.NO_RETURN:
                pass
            else:
                if verbose: print(result)

    def set_diagnostic(self, diagnostic):
        self.diagnostic = diagnostic

class ABCSampler(Sampler):
    _name = 'ABCSampler'
    """
    Approximate bayesian computation sampling from the posterior.  It uses the
    .simulate() method of the StochasticProcess object to get trajectories. It
    them checks to see if the end of the trajectory is within a tolerance
    factor of the observed ending point. If yes, it stores this trajectory to
    be returned later.
    """
    def __init__(self, tolerance, seed=0):
        super().__init__(seed)
        self.tolerance = tolerance

    def solve(self, stochastic_process, mc_samples, verbose=False):
        results = SamplingResults('ABCSampler', stochastic_process.true_trajectory)
        trajectories = []
        all_trajectories = []
        observed_ending_location = stochastic_process.xT
        starting_location = stochastic_process.x0
        for i in range(mc_samples):
            trajectory = stochastic_process.simulate(self.rng)
            final_position = trajectory[-1]
            # only accept this trajectory if it ended close to the final

            if self.tolerance == 'slacked':
                # TODO: find a way around this
                allowable = np.arange(2*stochastic_process.prior.start, 2*(stochastic_process.prior.start + stochastic_process.prior.n_numbers), 2)
                if np.sum(np.abs(observed_ending_location - final_position)) in allowable:
                    trajectories.append(trajectory)
            elif np.sum(np.abs(observed_ending_location - final_position)) <= self.tolerance:
                trajectories.append(trajectory)

            all_trajectories.append(trajectory)

            if self.diagnostic is not None:
                self.run_diagnostic(
                    SamplingResults.from_information(self._name, all_trajectories, trajectories),
                    verbose=verbose)

        results.all_trajectories(all_trajectories)
        results.trajectories(trajectories)
        results.create_posterior()
        return results


class MCSampler(Sampler):
    _name = 'MCSampler'
    def __init__(self, log_prob_tolerance=-10**10, seed=0):
        super().__init__(seed)
        # self.start_state = start_state
        self.log_prob_tolerance = log_prob_tolerance

    def draw_step(self, current_state, time_to_end=None, n_agents=1):
        steps_idx = self.rng.multinomial(1, self.step_probs, n_agents).argmax(axis=1)
        steps_taken = np.take(self.step_sizes, steps_idx, axis=0)
        step_log_probs = np.log(np.take(self.step_probs, steps_idx, axis=0)).sum()
        return steps_idx, steps_taken, step_log_probs

    def solve(self, stochastic_process, mc_samples, verbose=False):
        results = SamplingResults('MCSampler', stochastic_process.true_trajectory)
        self.step_probs, self.step_sizes = stochastic_process.step_probs, stochastic_process.step_sizes
        trajectories = []
        all_trajectories = []

        for i in range(mc_samples):
            x_t = stochastic_process.reset()  # start at the end
            trajectory_i = [x_t]
            log_path_prob = 0

            # go in reverse time:
            done = False
            while not done:
                x_t = trajectory_i[-1]
                # this is p(w_{t} | w_{t+1})
                step_idx, step, proposal_log_prob = self.draw_step(
                    x_t, n_agents=stochastic_process.n_agents)

                # reverse should be True here
                # see discussion: ./issues/11#issuecomment-379937140
                x_t, path_log_prob, done, _ = stochastic_process.step(
                    step_idx, reverse=True)

                # probability of the path gets updated:
                log_path_prob += path_log_prob
                # take the reverse step:
                trajectory_i.append(x_t)

            selected_trajectories = np.where(log_path_prob > -np.inf)

            sampled_trajectories = np.hstack(trajectory_i)[:, ::-1]

            sampled_trajectories = sampled_trajectories.reshape(
                stochastic_process.n_agents,
                stochastic_process.T,
                stochastic_process.dimensions)

            for traj_idx in selected_trajectories[0]:
                trajectories.append(
                    sampled_trajectories[traj_idx, :, :stochastic_process.dimensions])
            for m in range(sampled_trajectories.shape[0]):
                all_trajectories.append(sampled_trajectories[m, :, :stochastic_process.dimensions])

            if self.diagnostic is not None:
                self.run_diagnostic(
                    SamplingResults.from_information(
                        self._name, all_trajectories, trajectories),
                    other_information={
                        'override_count': i * stochastic_process.n_agents
                    }, verbose=verbose)

        results.all_trajectories(all_trajectories)
        results.trajectories(trajectories)
        results.create_posterior()
        return results


class ISSampler(Sampler):
    _name = 'ISSampler'
    def __init__(self, proposal, seed=0):
        super().__init__(seed)
        self.proposal = proposal
        self.soft = proposal._soft

    def solve(self, stochastic_process, mc_samples, verbose=False):
        results = ImportanceSamplingResults('ISSampler', stochastic_process.true_trajectory)

        proposal = self.proposal
        proposal.set_rng(self.rng)

        trajectories = []
        posterior_particles = []
        posterior_weights = []
        all_trajectories = []
        observed_ending_location = stochastic_process.xT
        x_0 = stochastic_process.x0

        for i in range(mc_samples):
            x_t = stochastic_process.reset()  # start at the end
            trajectory_i = [x_t]
            log_path_prob = 0.0
            log_proposal_prob = 0.0
            # go in reverse time:
            done = False
            while not done:
                x_t = trajectory_i[-1]
                # draw a reverse step
                # this is p(w_{t} | w_{t+1})
                step_idx, log_prob_proposal_step = proposal.draw(x_t, stochastic_process.transitions_left)
                # The IS proposal is already giving steps that go BACK in time
                # therefore we don't need to specify the reverse parameter
                # since it is actually already picking that reversed step.
                x_t, path_log_prob, done, _ = stochastic_process.step(step_idx, reverse=False)

                # accumulate log probs of the path and the proposal:
                log_path_prob += path_log_prob.reshape(-1)
                log_proposal_prob += log_prob_proposal_step.reshape(-1)

                trajectory_i.append(x_t)

            likelihood_ratio = log_path_prob - log_proposal_prob

            selected_trajectories = np.where(log_path_prob > -np.inf)

            sampled_trajectories = np.hstack(trajectory_i)[:, ::-1]

            sampled_trajectories = sampled_trajectories.reshape(
                stochastic_process.n_agents,
                stochastic_process.T,
                stochastic_process.dimensions)

            # TODO(zaf): Batchify these operations?
            for traj_idx in selected_trajectories[0]:
                trajectories.append(
                    sampled_trajectories[traj_idx, :, :stochastic_process.dimensions])
                posterior_particles.append(trajectories[-1][0])
                posterior_weights.append(np.exp(likelihood_ratio[traj_idx]))

            for m in range(sampled_trajectories.shape[0]):
                all_trajectories.append(sampled_trajectories[m, :, :stochastic_process.dimensions])

            if self.diagnostic is not None:
                self.run_diagnostic(
                    ImportanceSamplingResults.from_information(
                        self._name, all_trajectories, trajectories, posterior_particles, posterior_weights),
                    other_information={
                        'override_count': i * stochastic_process.n_agents
                    }, verbose=verbose)

        results.all_trajectories(all_trajectories)
        results.trajectories(trajectories)
        results.posterior_weights(posterior_weights)
        results.posterior(posterior_particles)
        return results
