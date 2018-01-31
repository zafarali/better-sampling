import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Samplers import ISSampler, ABCSampler, MCSampler
from StochasticProcess import RandomWalk
from proposal_distributions import MinimalProposal
from plotting import plot_mean_trajectories
sns.set_style('white')

MC_SAMPLES = 1000
POSSIBLE_STEPS = [[-1, -1], [1, 1], [1, -1], [0, 1], [0, 0]]
STEP_PROBS = np.ones(5)/5

# first simulate a random walk
rw = RandomWalk(2, STEP_PROBS, POSSIBLE_STEPS, n_agents=1, T=100)
rw.reset()

# run importance sampling
iss = ISSampler(MinimalProposal, log_prob_tolerance=-1.40*10**3)
iss_trajectories = iss.solve(rw, MC_SAMPLES)

# run ABC sampling
abc = ABCSampler(2)
abc_trajectories = abc.solve(rw, MC_SAMPLES)

# run MC Sampling
mcs = MCSampler(log_prob_tolerance=-1.4*10**3)
mcs_trajectories = mcs.solve(rw, MC_SAMPLES)



f = plt.figure(figsize=(8, 12))
ax = f.add_subplot(311)
ax = plot_mean_trajectories(iss_trajectories, np.arange(rw.T),rw.true_trajectory, ax=ax)
ax.set_title('{}:% successful {}'.format('IS', 100*len(iss_trajectories)/MC_SAMPLES))
ax = f.add_subplot(312)
ax = plot_mean_trajectories(abc_trajectories, np.arange(rw.T),rw.true_trajectory, ax=ax)
ax.set_title('{}:% successful {}'.format('ABC', 100*len(abc_trajectories)/MC_SAMPLES))
ax = f.add_subplot(313)
ax = plot_mean_trajectories(mcs_trajectories, np.arange(rw.T), rw.true_trajectory, ax=ax)
ax.set_title('{}:% successful {}'.format('MCS', 100*len(mcs_trajectories)/MC_SAMPLES))
f.set_tight_layout(True)
f.savefig('method_comparison_figure.pdf')