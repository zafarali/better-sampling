import pg_methods.utils.gradients as gradients
from torch.nn.utils import clip_grad_norm
import torch
from torch.autograd import Variable
from pg_methods.utils.data import MultiTrajectory
from pg_methods.utils.objectives import PolicyGradientObjective
import numpy as np
from .Samplers import Sampler
from ..results import RLSamplingResults
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
                 use_cuda=False):
        """
        The reinforced variational inference sampler
        :param policy: the policy to use
        :param policy_optimizer: the optimizer for the policy
        :param baseline: a baseline function
        :param feed_time: a boolean indicating if we should or
                          should not feed time into the proposal
        :param seed: the seed to use
        :param use_cuda: uses cuda (currently not working)
        """
        Sampler.__init__(self, seed)
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.baseline = baseline
        self.use_cuda = use_cuda
        self._training = True
        self.feed_time = feed_time
        self.objective = objective

    def train_mode(self, mode):
        self._training = mode
        logging.warning('Train mode has been changed to {}. Make sure to also update the PyTorchWrap train_mode as well..'.format(self._training))

    def solve(self, stochastic_process, mc_samples, verbose=False):
        feed_time = self.feed_time
        assert stochastic_process._pytorch, 'Your stochastic process must be pytorch wrapped.'
        results = RLSamplingResults('RVISampler', stochastic_process.true_trajectory)
        trajectories = []
        observed_ending_location = stochastic_process.xT
        x_0 = stochastic_process.x0
        rewards_per_episode = []
        loss_per_episode = []
        all_trajectories = []
        posterior_particles = []
        posterior_weights = []
        for i in range(mc_samples):
            x_t = stochastic_process.reset()  # start at the end
            x_tm1 = x_t
            trajectory_i = [x_t.data.cpu().numpy() if isinstance(x_t, Variable) else x_t.cpu().numpy()]

            if feed_time:
                # this will augment the state space with the time dimension
                # so that the learner has access to it.
                x_with_time = torch.zeros(stochastic_process.n_agents, x_tm1.size()[-1]+1)
                x_with_time[:, :x_tm1.size()[-1]].copy_(x_tm1.data)
                x_with_time[:, x_tm1.size()[-1]].copy_(torch.ones(stochastic_process.n_agents)) # 100% of the time is left
                x_tm1 = Variable(x_with_time)

            log_path_prob = np.zeros((stochastic_process.n_agents, 1))
            log_proposal_prob = np.zeros((stochastic_process.n_agents, 1))
            policy_gradient_trajectory_info = MultiTrajectory(stochastic_process.n_agents)

            done = torch.IntTensor([0])
            while not np.all(done.numpy()):
                # draw a reverse step
                # this is p(w_{t} | w_{t+1})
                assert len(x_tm1.size()) == 2
                action,  log_prob_action = self.policy(x_tm1)
                # print('proposal_log_prob step:',log_prob_proposal_step)
                x_t, path_log_prob, done, _ = stochastic_process.step(action, reverse=False)


                reward_ = path_log_prob.float().view(-1,1) - log_prob_action.data.float().view(-1, 1)

                reward = torch.zeros_like(reward_)
                reward.copy_(reward_)
                reward[reward <= -np.inf] = -100000. # throw away infinite negative rewards
                # if t==0: print(reward)
                value_estimate = self.baseline(x_t) if self.baseline is not None else torch.FloatTensor([[0]*stochastic_process.n_agents])

                # probability of the path gets updated:
                log_path_prob += path_log_prob.numpy().reshape(-1, 1)
                log_proposal_prob += log_prob_action.data.cpu().float().numpy().reshape(-1, 1)
                # take the reverse step:
                # if isinstance()
                if isinstance(x_t, Variable):
                    trajectory_i.append(x_t.data.numpy())
                else:
                    trajectory_i.append(x_t.numpy())

                if feed_time:
                    # this will augment the state space with the time dimension
                    # so that the learner has access to it.
                    x_with_time = torch.zeros(stochastic_process.n_agents, x_tm1.size()[-1])
                    x_with_time[:, :x_tm1.size()[-1]-1].copy_(x_t.data)
                    x_with_time[:, x_tm1.size()[-1]-1].copy_((stochastic_process.transitions_left/(stochastic_process.T-1)) * torch.ones(stochastic_process.n_agents))
                    x_t = Variable(x_with_time)
                    assert x_tm1.size() == x_t.size(), 'State sizes must match, but they dont. {} != {}'.format(x_tm1.size(), x_t.size())
                    policy_gradient_trajectory_info.append(x_tm1[:, :-1], action, reward, value_estimate,
                                                           log_prob_action, x_t[:, :-1], done)
                else:
                    policy_gradient_trajectory_info.append(x_tm1, action, reward, value_estimate, log_prob_action, x_t, done)

                x_tm1 = x_t

                # endwhile loop

            policy_gradient_trajectory_info.torchify()

            # update the proposal distribution
            if self._training:
                returns = gradients.calculate_returns(policy_gradient_trajectory_info.rewards, 1, None)
                advantages = returns - policy_gradient_trajectory_info.values
                if self.baseline is not None:
                    self.baseline.update_baseline(policy_gradient_trajectory_info, returns)

                loss = self.objective(advantages, policy_gradient_trajectory_info)

                if self.use_cuda:
                    loss = loss.cuda()

                if self.policy_optimizer is not None:
                    self.policy_optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm(self.policy.fn_approximator.parameters(), 40)
                    self.policy_optimizer.step()
                # end training statement.

            reward_summary = torch.sum(policy_gradient_trajectory_info.rewards, dim=0).mean()
            rewards_per_episode.append(reward_summary)
            if self._training: loss_per_episode.append(loss.cpu().data[0])
            if i % 100 == 0 and verbose and self._training:
                print('MC Sample {}, loss {:3g}, episode_reward {:3g}, successful trajs {}'.format(i, loss.cpu().data[0], reward_summary, len(trajectories)))


            if feed_time:
                trajectory_i = np.hstack(trajectory_i).reshape(stochastic_process.n_agents, stochastic_process.T,
                                                               x_t.size()[-1]-1)
            else:
                trajectory_i = np.hstack(trajectory_i).reshape(stochastic_process.n_agents, stochastic_process.T,
                                                               x_t.size()[-1])

            # select paths for storage
            log_proposal_prob += np.log(stochastic_process.prior_pdf(x_t))
            likelihood_ratios = log_path_prob - log_proposal_prob
            selected_trajectories = np.where(log_path_prob > -np.inf)
            for traj_idx in selected_trajectories[0]:
                    trajectories.append(trajectory_i[traj_idx, ::-1, :stochastic_process.dimensions])
                    posterior_particles.append(trajectories[-1][0])
                    posterior_weights.append(np.exp(likelihood_ratios[traj_idx]))
            for m in range(trajectory_i.shape[0]):
                all_trajectories.append(trajectory_i[m, ::-1, :stochastic_process.dimensions])

        results.all_trajectories(all_trajectories)
        results.trajectories(trajectories)
        results.posterior(posterior_particles)
        results.posterior_weights(posterior_weights)
        if self._training: results.loss_per_episode = loss_per_episode
        results.rewards_per_episode = rewards_per_episode
        return results
