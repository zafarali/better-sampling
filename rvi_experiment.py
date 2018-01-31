import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from StochasticProcess import RandomWalk, PyTorchWrap
from RVISampler import RVISampler
from plotting import plot_mean_trajectories, plot_trajectory_time_evolution
import torch
import torch.nn as nn
from pg_methods.utils.baselines import MovingAverageBaseline
from pg_methods.utils.policies import MultinomialPolicy, RandomPolicy
from pg_methods.utils.networks import MLP_factory

MC_SAMPLES = 1000
POSSIBLE_STEPS = np.array([[-1, -1], [1, 1], [1, -1], [0, 1], [0, 0]])
STEP_PROBS = np.ones(5)/5
# POSSIBLE_STEPS = np.array([[-1], [+1], [0]])
# STEP_PROBS = np.ones(3)/3
DIMENSIONS = POSSIBLE_STEPS.shape[1]
# first simulate a random walk
rw = PyTorchWrap(RandomWalk(DIMENSIONS, STEP_PROBS, POSSIBLE_STEPS, n_agents=10, T=100))
rw.reset()

# real policy
fn_approximator = MLP_factory(DIMENSIONS, hidden_sizes=[32, 32], output_size=POSSIBLE_STEPS.shape[0], hidden_non_linearity=nn.ReLU)
policy = MultinomialPolicy(fn_approximator)
policy_optimizer = torch.optim.RMSprop(fn_approximator.parameters(),lr=0.001)

# random policy

# policy = RandomPolicy(POSSIBLE_STEPS.shape[0])
# policy_optimizer = None


rvi = RVISampler(policy, policy_optimizer, baseline=MovingAverageBaseline(0.99), log_prob_tolerance=-2*10**2)
rvis_trajectories, rvis_losses, rvis_rewards, all_trajectories = rvi.solve(rw, MC_SAMPLES, verbose=True)

sns.set_style('white')
f = plt.figure(figsize=(9, 4))
ax = f.add_subplot(121)
ax = plot_mean_trajectories(rvis_trajectories, np.arange(rw.T), rw.true_trajectory, ax=ax)
ax.set_title('Reinforced Variational Inference % success {0:1g}'.format(100*len(rvis_trajectories)/(MC_SAMPLES*10)))
ax = f.add_subplot(122)
ax = plot_trajectory_time_evolution(all_trajectories, 1, step=200, ax=ax)
ax.set_title('Evolution of Trajectories over MCSamples \n (lighter = more recent, dim 1)')
f.tight_layout()
f.save_fig('RVI performance')


# def downsample(array, step=50):
#     to_return = []
#     steps = []
#     for i in range(0, array.shape[0], step):
#         to_return.append(array[i])
#         steps.append(i)
#
#     return np.array(steps), np.array(to_return)
#
#
# plt.plot(*downsample(np.array(rvis_rewards)))
# # plt.ylim([0, -650])