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


"""
Experiment without time information
"""
# real policy
fn_approximator = MLP_factory(DIMENSIONS, hidden_sizes=[32, 32], output_size=POSSIBLE_STEPS.shape[0], hidden_non_linearity=nn.ReLU)
policy = MultinomialPolicy(fn_approximator)
policy_optimizer = torch.optim.RMSprop(fn_approximator.parameters(),lr=0.001)



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
f.suptitle('Without time information')
f.savefig('RVIperformance_without_time.pdf')



"""
Experiment without time information
"""
fn_approximator = MLP_factory(DIMENSIONS+1, hidden_sizes=[32, 32], output_size=POSSIBLE_STEPS.shape[0], hidden_non_linearity=nn.ReLU)
policy = MultinomialPolicy(fn_approximator)
policy_optimizer = torch.optim.RMSprop(fn_approximator.parameters(),lr=0.001)


rvi = RVISampler(policy, policy_optimizer, baseline=MovingAverageBaseline(0.9), log_prob_tolerance=-2*10**2)
rvis_trajectories, rvis_losses, rvis_rewards, all_trajectories = rvi.solve(rw, MC_SAMPLES, verbose=True, feed_time=True)

sns.set_style('white')
f = plt.figure(figsize=(9, 4))
ax = f.add_subplot(121)
ax = plot_mean_trajectories(rvis_trajectories, np.arange(rw.T), rw.true_trajectory, ax=ax)
ax.set_title('Reinforced Variational Inference % success {0:1g}'.format(100*len(rvis_trajectories)/(MC_SAMPLES*10)))
ax = f.add_subplot(122)
ax = plot_trajectory_time_evolution(all_trajectories, 1, step=200, ax=ax)
ax.set_title('Evolution of Trajectories over MCSamples \n (lighter = more recent, dim 1)')
f.tight_layout()
f.suptitle('With time information')
f.savefig('RVIperformance_wwithtime.pdf')

