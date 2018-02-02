import pg_methods.utils.gradients as gradients
from torch.nn.utils import clip_grad_norm
import torch
from torch.autograd import Variable
from pg_methods.utils.data import MultiTrajectory
import numpy as np
from Samplers import Sampler

class RVISampler(Sampler):
    """
    Reinforced Variational Inference Sampler
    """
    def __init__(self, policy, policy_optimizer, baseline=None,  log_prob_tolerance=-10**10, seed=0, use_cuda=False):
        Sampler.__init__(self, seed)
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.baseline = baseline
        self.use_cuda = use_cuda
        self.log_prob_tolerance = log_prob_tolerance

    def solve(self, stochastic_process, mc_samples, verbose=False, feed_time=False):
        assert stochastic_process._pytorch, 'Your stochastic process must be pytorch wrapped.'

        trajectories = []
        observed_ending_location = stochastic_process.xT
        x_0 = stochastic_process.x0
        rewards_per_episode = []
        loss_per_episode = []
        all_trajectories = []
        for i in range(mc_samples):
            x_t = stochastic_process.reset()  # start at the end
            x_tm1 = x_t
            trajectory_i = [x_t.data.cpu().numpy() if isinstance(x_t, Variable) else x_t.cpu().numpy()]

            if feed_time:
                # this will augment the state space with the time dimension
                # so that the learner has access to it.
                x_with_time = torch.zeros(stochastic_process.n_agents, x_tm1.size()[-1]+1)
                x_with_time[:, :x_tm1.size()[-1]].copy_(x_tm1.data)
                x_with_time[:, x_tm1.size()[-1]].copy_(torch.ones(stochastic_process.n_agents))
                x_tm1 = Variable(x_with_time)

            log_path_prob = np.zeros((stochastic_process.n_agents, 1))
            policy_gradient_trajectory_info = MultiTrajectory(stochastic_process.n_agents)
            # go in reverse time:
            for t in reversed(range(0, stochastic_process.T-1)):
                # draw a reverse step
                # this is p(w_{t} | w_{t+1})
                action,  log_prob_action = self.policy(x_tm1)
                # print('proposal_log_prob step:',log_prob_proposal_step)
                x_t, path_log_prob, done, _ = stochastic_process.step(action)

                if t != 0: # until we reach time 0
                    # provide an "instant reward"
                    reward = path_log_prob.float() - log_prob_action.data.float().view(-1, 1)
                else:
                    reward = path_log_prob.float()

                # print(reward)
                reward[reward < -10**10] = -10000. # throw away infinite negative rewards

                value_estimate = self.baseline(x_t) if self.baseline is not None else torch.FloatTensor([[0]*stochastic_process.n_agents])

                # probability of the path gets updated:
                log_path_prob += path_log_prob.numpy()
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
                    x_with_time[:, x_tm1.size()[-1]-1].copy_((t/stochastic_process.T) * torch.ones(stochastic_process.n_agents))
                    x_t = Variable(x_with_time)
                    assert x_tm1.size() == x_t.size(), 'State sizes must match, but they dont. {} != {}'.format(x_tm1.size(), x_t.size())
                    policy_gradient_trajectory_info.append(x_tm1[:, :-1], action, reward, value_estimate,
                                                           log_prob_action, x_t[:, :-1], done)
                else:
                    policy_gradient_trajectory_info.append(x_tm1, action, reward, value_estimate, log_prob_action, x_t, done)

                x_tm1 = x_t

            # print(trajectory_i)
            # conduct an update step.
            # print(policy_gradient_trajectory_info.rewards)
            policy_gradient_trajectory_info.torchify()
            returns = gradients.calculate_returns(policy_gradient_trajectory_info.rewards, 1, None)
            advantages = returns - policy_gradient_trajectory_info.values
            if self.baseline is not None:
                self.baseline.update_baseline(policy_gradient_trajectory_info.rewards,
                                              advantages,
                                              policy_gradient_trajectory_info.values)

            loss = gradients.calculate_policy_gradient_terms(policy_gradient_trajectory_info.log_probs, advantages)
            loss = loss.sum(dim=0).mean()
            if self.use_cuda:
                loss = loss.cuda()

            if self.policy_optimizer is not None:
                self.policy_optimizer.zero_grad()
                loss.backward()
                clip_grad_norm(self.policy.fn_approximator.parameters(), 40)
                self.policy_optimizer.step()

            reward_summary = torch.sum(policy_gradient_trajectory_info.rewards, dim=0).mean()
            rewards_per_episode.append(reward_summary)
            loss_per_episode.append(loss.cpu().data[0])
            if i % 100 == 0 and verbose:
                print('MC Sample {}, loss {:3g}, episode_reward {:3g}, successful trajs {}'.format(i, loss.cpu().data[0], reward_summary, len(trajectories)))

            if feed_time:
                trajectory_i = np.hstack(trajectory_i).reshape(stochastic_process.n_agents, stochastic_process.T + 1,
                                                               x_t.size()[-1]-1)
            else:
                trajectory_i = np.hstack(trajectory_i).reshape(stochastic_process.n_agents, stochastic_process.T + 1,
                                                               x_t.size()[-1])

            selected_trajectories = np.where(log_path_prob > -np.inf)
            for traj_idx in selected_trajectories[0]:
                    trajectories.append(trajectory_i[traj_idx, ::-1, :stochastic_process.dimensions])
            for m in range(trajectory_i.shape[0]):
                all_trajectories.append(trajectory_i[m, ::-1, :stochastic_process.dimensions])
        return trajectories, loss_per_episode, rewards_per_episode, all_trajectories
