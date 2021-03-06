from torch.nn.utils import clip_grad_norm_
import torch
from pg_methods.data import MultiTrajectory
from pg_methods.objectives import PolicyGradientObjective
from pg_methods import interfaces
import pg_methods.gradients as gradients
import numpy as np
from .Samplers import Sampler
from ..results import RLSamplingResults
from rvi_sampling import utils
import logging

class RVISampler(Sampler):
    """
    Reinforced Variational Inference Sampler
    """
    _name = 'RVISampler'
    def __init__(self, policy,
                 policy_optimizer,
                 baseline=None,
                 feed_time=False,
                 objective=PolicyGradientObjective(),
                 seed=0,
                 gamma=1,
                 lam=1,
                 negative_reward_clip=-10,
                 use_cuda=False,
                 lr_scheduler=None,
                 use_gae=False,
                 multitask_training=False,
                 reward_type='KLD'):
        """
        The reinforced variational inference sampler
        :param policy: the policy to use
        :param policy_optimizer: the optimizer for the policy
        :param baseline: a baseline function
        :param feed_time: a boolean indicating if we should or
                          should not feed time into the proposal
        :param seed: the seed to use
        :param use_cuda: uses cuda (currently not working)
        :param multitask_training: boolean indicating if a new task should be
            sampled before every training step.
        :param reward_type: A string representing the type of reward to give
            to the reinforcement learning algorithm. Currently supports
            KL divergence: 'KLD' and Chi-squared divergence: 'C2'.
        """
        Sampler.__init__(self, seed)
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.baseline = baseline
        self.use_cuda = use_cuda
        self._training = True
        self.feed_time = feed_time
        self.objective = objective
        self.gamma = gamma
        self.negative_reward_clip = negative_reward_clip
        self.lr_scheduler=lr_scheduler
        self.train_steps_completed = 0
        self._reward_implementation = reward_type
        self._reward_buffer = []

        # Generalized Advantages
        self.use_gae = use_gae
        self.lam = lam

        # Training Details
        self.multitask_training = multitask_training

    def train_mode(self, mode):
        self._training = mode

    def solve(self, stochastic_process, mc_samples, verbose=False):
        """
        Collect mc_samples from the posterior.
        Note that this function will first train the proposal before evaluating
        This is a departure from version 0.0.0 where samples were collected at
        the same time as the training. In theory both of these could be useful
        in different cases (online learning setting)
        :param stochastic_process: The stochastic process to learn from
        :param mc_samples: The number of samples to obtain
        :param verbose: verbosity

        :return:
        """

        with torch.set_grad_enabled(self._training):
            results = self.train(stochastic_process, mc_samples, verbose=verbose)

        return results


    def train(self, stochastic_process, train_episodes, verbose=False):
        """
        Train the RVI model
        :param stochastic_process: The stochastic process to learn from
        :param train_steps: The total number of steps to train for
        :param verbose:
        :return:
        """

        self.check_stochastic_process(stochastic_process)

        results = RLSamplingResults('RVISampler', stochastic_process.true_trajectory)

        observed_ending_location = stochastic_process.xT
        x_0 = stochastic_process.x0

        # tracks the
        rewards_per_episode = []
        loss_per_episode = []
        all_trajectories = []
        posterior_particles = []
        posterior_weights = []
        saved_trajectories = []

        for i in range(train_episodes):
            if self.multitask_training:
                stochastic_process.new_task()

            pg_info, sampled_trajectory, log_path_prob, log_proposal_prob = self.do_rollout(stochastic_process, verbose)

            returns = gradients.calculate_returns(pg_info.rewards, self.gamma, pg_info.masks)

            if self.use_gae:
                advantages = gradients.calculate_gae(
                    interfaces.pytorch2list(pg_info.rewards),
                    interfaces.pytorch2list(pg_info.values),
                    self.gamma,
                    self.lam)
            else:
                advantages = returns - pg_info.values

            pg_loss = self.objective(advantages, pg_info)

            if self.baseline is not None:
                if type(self.baseline).__name__ == "FunctionApproximatorBaseline":
                    val_loss = self.baseline.update_baseline(
                        pg_info,
                        returns,
                        tro=False)
                else:
                    val_loss = self.baseline.update_baseline(pg_info, returns)
            else:
                val_loss = 0

            loss = pg_loss

            if self._training:
                self.policy_optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.policy.fn_approximator.parameters(), 40) # TODO: what clipping value to use here?
                self.policy_optimizer.step()
                if self.lr_scheduler is not None: self.lr_scheduler.step()

            reward_summary = torch.sum(pg_info.rewards, dim=0).mean()
            rewards_per_episode.append(reward_summary)
            loss_per_episode.append(loss.item())

            if i % 100 == 0 and verbose:
                print('Train Step {}, loss {:3g}, episode_reward {:3g}, '
                      'trajectory_length {}, successful trajs {}, '
                      'path_log_prob: {}, proposal_log_prob: {}'.format(self.train_steps_completed,
                                                                        loss_per_episode[-1],
                                                                        reward_summary,
                                                                        len(sampled_trajectory),
                                                                        len(saved_trajectories),
                                                                        np.mean(log_path_prob),
                                                                        np.mean(log_proposal_prob)))
            
            # technically this is only to track if improvement is happening
            # we can make this optional because ideally most people don't care about
            # this except for diagnostic purposes
            # decide if the trajectory should be saved into the posterior.
            self.maybe_save_trajectory_into_posterior(stochastic_process, sampled_trajectory,
                                                      log_path_prob, log_proposal_prob,
                                                      posterior_particles, posterior_weights,
                                                      all_trajectories, saved_trajectories)

            # use information saved so far to run a diagnostic.
            if self.diagnostic is not None:
                self.run_diagnostic(
                    RLSamplingResults.from_information(
                        self._name, all_trajectories, saved_trajectories, posterior_particles, posterior_weights),
                    other_information={
                        'episode_reward': reward_summary.item(),
                        'trajectory_length': len(sampled_trajectory),
                        'path_log_prob': np.mean(log_path_prob),
                        'proposal_log_prob': np.mean(log_proposal_prob),
                        'override_count': self.train_steps_completed * stochastic_process.n_agents
                    }, verbose=verbose)
            self.train_steps_completed += 1

        results.all_trajectories(all_trajectories)
        results.trajectories(saved_trajectories)
        results.posterior(posterior_particles)
        results.posterior_weights(posterior_weights)
        results.loss_per_episode = loss_per_episode
        results.rewards_per_episode = rewards_per_episode
        return results

    def sample_from_posterior(self, stochastic_process, mc_samples, verbose=False):
        """
        Evaluates draws from the RVI model (with no training)
        :param stochastic_process: The stochastic process to learn from
        :param mc_samples: the number of draws to make from the proposal
        :param verbose:
        :return:
        """

        self.train_mode(False)
        self.check_stochastic_process(stochastic_process)
        stochastic_process.train_mode(False)

        # collect metrics
        results = RLSamplingResults('RVISampler', stochastic_process.true_trajectory)
        all_trajectories = []
        saved_trajectories = []
        posterior_particles = []
        posterior_weights = []

        for _ in range(mc_samples):
            _, sampled_trajectory, log_path_prob, log_proposal_prob = self.do_rollout(stochastic_process, verbose)
            # do something with the trajectory here.
            self.maybe_save_trajectory_into_posterior(stochastic_process, sampled_trajectory,
                                                      log_path_prob, log_proposal_prob,
                                                      posterior_particles, posterior_weights,
                                                      all_trajectories, saved_trajectories)

        results.all_trajectories(all_trajectories)
        results.trajectories(saved_trajectories)
        results.posterior(posterior_particles)
        results.posterior_weights(posterior_weights)

        return results


    def do_rollout(self, stochastic_process, verbose=False):
        """
        Does a rollout with the current proposal distribution
        :param stochastic_process:
        :param verbose:
        :return:
        """
        self.check_stochastic_process(stochastic_process)

        x_t = stochastic_process.reset()
        x_tm1 = x_t
        sampled_trajectory = [x_t.data.numpy()]

        if self.feed_time:
            x_tm1 = self.augment_time(stochastic_process, x_tm1, x_t, start=True)

        log_path_prob = np.zeros((stochastic_process.n_agents, 1))
        log_proposal_prob = np.zeros((stochastic_process.n_agents, 1))
        pg_trajectory_info = MultiTrajectory(stochastic_process.n_agents)
        done = torch.IntTensor([0])

        while not np.all(done.numpy()):
            # draw a reverse step
            # this is p(w_{t} | w_{t+1})
            assert len(x_tm1.size()) == 2
            action, log_prob_action = self.policy(x_tm1)

            """
            Reverse Mode
            reverse should be True because the RVI learneer doesn't know that it is taking steps in the reverse
            for the matching. 
            """
            x_t, path_log_prob, done, _ = stochastic_process.step(action, reverse=True)

            """ 
            Clip the reward directly: ie minus the log prob from the environment and then minus the log 
                                        prob of the path. After this clip the reward.

            Both Reward Clipping and Reverse Mode were evaluated and are shown here:
            https://docs.google.com/document/d/1OugyMJ4pPwiUW_3im9-exA5Ynj5O8eIv5GaVC6-laHc
            We found that (as expected) reverse True performed best. Unsurprisingly Option B
            performed best (as it is the theoretically justified method of doing this)

            Therefore, USE OPTION B AND REVERSE = TRUE 
            """

            # OPTION A: (kept here for historical purposes only)
            # reward_ = path_log_prob.float().view(-1,1)

            # OPTION B:
            reward = self._process_rewards(path_log_prob, log_prob_action, done)

            # OPTION A: (kept here for historical purposes only)
            # reward -= log_prob_action.data.float().view(-1, 1)

            # probability of the path gets updated:
            log_path_prob += path_log_prob.numpy().reshape(-1, 1)
            log_proposal_prob += log_prob_action.data.float().numpy().reshape(-1, 1)

            sampled_trajectory.append(x_t.data.numpy())

            if self.feed_time:
                x_t = self.augment_time(stochastic_process, x_tm1, x_t)
                assert x_tm1.size() == x_t.size(), 'State sizes before and after must match, ' \
                                                   'but they dont. {} != {}'.format(x_tm1.size(), x_t.size())

            # TODO(zaf): check baseline calculations here.
            if self.baseline is not None:
                value_estimate = self.baseline(x_t)
            else:
                value_estimate = torch.from_numpy(np.zeros(stochastic_process.n_agents, 1)).float()

            pg_trajectory_info.append(x_tm1, action, reward, value_estimate, log_prob_action, x_t, done)

            x_tm1 = x_t


        pg_trajectory_info.torchify()

        return pg_trajectory_info, sampled_trajectory, log_path_prob, log_proposal_prob

    def _process_rewards(self, path_log_prob, log_prob_action, done=None):
        reward_ = (
                path_log_prob.float().view(-1, 1) -
                log_prob_action.data.float().view(-1, 1)
        )
        reward = torch.zeros_like(reward_)
        reward.copy_(reward_)

        if self._reward_implementation == 'KLD':
            if self.negative_reward_clip is not None:
                # throw away infinite negative rewards
                reward[reward <= -np.inf] = float(self.negative_reward_clip)
            return reward
        elif self._reward_implementation== 'C2':
            if done is None: raise ValueError('Done is required for C2.')
            self._reward_buffer.append(reward)
            if np.all(done.numpy()):
                traj_ratio = torch.stack(self._reward_buffer).sum(0).exp() ** 2
                self._reward_buffer = []
                return -traj_ratio.view_as(reward_)
            else:
                # Delay the rewards until the end of the trajectory.
                return torch.zeros_like(reward_)
            raise NotImplementedError
        else:
            raise ValueError(
                    'Unknown objective function {}.'
                    'Must be C2 or KLD'.format(self.objective_function))

    # UTILITY FUNCTIONS
    def check_stochastic_process(self, stochastic_process):
        # checks if a stochastic process is pytorch wrapped.
        if not stochastic_process._pytorch:
            raise ValueError('Stochastic Process must be pytorch wrapped.')

    def augment_time(self, stochastic_process, x_tm1, x_t, start=False):
        with torch.set_grad_enabled(not self._training):
            # if this is the first iteration, expansion of dimensions is necessary
            expand_dims = int(start)
            contract_dims = int(not start)
            # this will augment the state space with the time dimension
            # so that the learner has access to it.

            # add another dimension to the state space dimension
            x_with_time = torch.zeros(stochastic_process.n_agents, x_tm1.size()[-1]+expand_dims)
            # fill in all but the last entry with the state information
            x_with_time[:, :x_tm1.size()[-1] - contract_dims].copy_(x_t.data)

            # fill in the last entry with the time information
            proportion_of_time_left = (stochastic_process.transitions_left / (stochastic_process.T - 1))
            x_with_time[:, x_tm1.size()[-1] - contract_dims].copy_(proportion_of_time_left * torch.ones(stochastic_process.n_agents))

            # create a new variable with this information
            return x_with_time

    def maybe_save_trajectory_into_posterior(self, stochastic_process,
                                             sampled_trajectories,
                                             log_path_prob, log_proposal_prob,
                                             posterior_particles, posterior_weights,
                                             all_trajectories, saved_trajectories):
        """
        Decides if the trajectory can be saved into the posterior and saves it.
        :param stochastic_process: The stochastic process
        :param sampled_trajectories: The sampled trajectories
        :param log_path_prob: The log prob of the paths
        :param log_proposal_prob: The log prob of the proposal
        :param posterior_particles: The posterior particles
        :param posterior_weights: the posterior weights
        :param all_trajectories: all trajectories
        :param saved_trajectories: the trajectories that are saved
        :return:
        """
        # TODO: replace stochastic_process.dimensions with stochastic_process.state_space?

        # stack and reshape the collected trajectories so that we can
        # apply numpy operations to it
        reshape_last_dim = stochastic_process.dimensions
        sampled_trajectories = np.hstack(sampled_trajectories).reshape(stochastic_process.n_agents,
                                                                       len(sampled_trajectories),
                                                                       reshape_last_dim)
        # Decide which paths are fit for storage
        likelihood_ratios = log_path_prob - log_proposal_prob
        selected_trajectories = np.where(log_path_prob > -np.inf)

        # TODO: look for ways to get rid of this for loop?
        for traj_idx in selected_trajectories[0]:
            saved_trajectories.append(sampled_trajectories[traj_idx, ::-1, :stochastic_process.dimensions])
            posterior_particles.append(saved_trajectories[-1][0])
            posterior_weights.append(np.exp(likelihood_ratios[traj_idx]))
        for m in range(sampled_trajectories.shape[0]):
            all_trajectories.append(sampled_trajectories[m, ::-1, :stochastic_process.dimensions])


