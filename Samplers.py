class Sampler(object):
    """
    Abstract class that sample from Posteriors 
    """
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
    def solve(self, stochastic_process, mc_samples):
        pass


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

