import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rvi_sampling.samplers import ISSampler, ABCSampler, MCSampler, RVISampler
from rvi_sampling.StochasticProcess import RandomWalk
from rvi_sampling.distributions.proposal_distributions import MinimalProposal, SimonsProposal
from rvi_sampling.distributions.prior_distributions import DiscreteUniform
from rvi_sampling.plotting import plot_mean_trajectories, plot_trajectory_time_evolution
sns.set_style('white')

MC_SAMPLES = 2000
POSSIBLE_STEPS = [[-1], [+1]]
STEP_PROBS = np.ones(2)/2
DIMENSIONS = 1
# first simulate a random walk
rw = RandomWalk(DIMENSIONS, STEP_PROBS, POSSIBLE_STEPS, n_agents=1, T=100, prior_distribution=DiscreteUniform(DIMENSIONS, -2, 4))
rw.reset()

samplers = [ISSampler(), ABCSampler(), MCSampler(), RVISampler() ]

print('True Starting Position is:{}'.format(rw.x0))
# run importance sampling
iss = ISSampler(SimonsProposal)
iss_results = iss.solve(rw, MC_SAMPLES)
iss_trajectories, iss_all_traj = iss_results.trajectories(), iss_results.all_trajectories()
print('IS Estimate: {} | variance {}'.format(iss_results.expectation(True), iss_results.variance(True)))
# run ABC sampling
abc = ABCSampler(2)
abc_results = abc.solve(rw, MC_SAMPLES)
abc_trajectories, abc_all_traj = abc_results.trajectories(), abc_results.all_trajectories()
print('ABC Estimate: {} | variance {}'.format(abc_results.expectation(False), abc_results.variance(False)))
# run MC Sampling
mcs = MCSampler()
mc_results = mcs.solve(rw, MC_SAMPLES)
mcs_trajectories, mcs_all_traj = mc_results.trajectories(), mc_results.all_trajectories()
print('MC Estimate: {} | variance {}'.format(mc_results.expectation(False), mc_results.variance(False)))

f = plt.figure(figsize=(8, 12))
ax = f.add_subplot(331)
ax = plot_mean_trajectories(iss_trajectories, np.arange(rw.T),rw.true_trajectory, ax=ax)
ax.set_title('{}:% successful {}'.format('IS', 100*len(iss_trajectories)/MC_SAMPLES))
ax = f.add_subplot(332)
ax = plot_trajectory_time_evolution(iss_all_traj, 0, step=200, ax=ax)
ax = f.add_subplot(333)
ax = plot_trajectory_time_evolution(iss_trajectories, 0, step=1, ax=ax)
ax = f.add_subplot(334)
ax = plot_mean_trajectories(abc_trajectories, np.arange(rw.T),rw.true_trajectory, ax=ax)
ax.set_title('{}:% successful {}'.format('ABC', 100*len(abc_trajectories)/MC_SAMPLES))
ax = f.add_subplot(335)
ax = plot_trajectory_time_evolution(abc_all_traj, 0, step=200, ax=ax)
ax = f.add_subplot(336)
ax = plot_trajectory_time_evolution(abc_trajectories, 0, step=1, ax=ax)
ax = f.add_subplot(337)
ax = plot_mean_trajectories(mcs_trajectories, np.arange(rw.T), rw.true_trajectory, ax=ax)
ax.set_title('{}:% successful {}'.format('MCS', 100*len(mcs_trajectories)/MC_SAMPLES))
ax = f.add_subplot(338)
ax = plot_trajectory_time_evolution(mcs_all_traj, 0, step=200, ax=ax)
ax = f.add_subplot(339)
ax = plot_trajectory_time_evolution(mcs_trajectories, 0, step=1, ax=ax)
f.set_tight_layout(True)
f.savefig('method_comparison_figure.pdf')