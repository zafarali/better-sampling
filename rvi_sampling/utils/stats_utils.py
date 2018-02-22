import numpy as np

def bootstrap_idx(dataset_size, n_bootstraps=150):
    """
    Obtains indices for bootstrapping
    :param dataset_size: size of the dataset
    :param n_bootstraps: number of bootstraps to run
    :return:
    """
    data_idx = np.random.choice(np.arange(dataset_size), size=(n_bootstraps, dataset_size), replace=True)
    return data_idx

def bootstrap_data(particles, weights, hist_range, n_bootstraps, percentile=5):
    """
    Calculates the bootstrap for the error bars in a histogram

    Example usage to plot error bars of a histogram
    ```
    delta05, delta95, vals_ = bootstrap_data(data['posterior_particles'],
                                          data['posterior_weights'],
                                          hist_range,
                                          10000,
                                        percentile=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    probs, _, _ = ax.hist(data['posterior_particles'],
                           bins=hist_range,
                           density=True,
                           weights=data['posterior_weights'],
                           alpha=0.7)
    ebars = np.stack([probs+delta05, probs+delta95])
    eb = ax.errorbar(vals_, probs, yerr=ebars, color='k', linestyle='none')
    ```
    :param particles: the particles
    :param weights: the weights of each particle
    :param hist_range: the range of the histogram
    :param n_bootstraps: the number of boostraps to use
                         in the estimation process
    :param percentile: the CI percentile to return
    :return:
    """
    particles = np.array(particles)
    weights = np.array(weights)
    hist_probs, vals = np.histogram(particles, bins=hist_range, density=True, weights=weights)

    prob_estimates = []
    for idx in bootstrap_idx(particles.shape[0], n_bootstraps):
        probs, vals = np.histogram(particles[idx], bins=hist_range, density=True, weights=weights[idx])
        prob_estimates.append(probs)

    prob_estimates = np.stack(prob_estimates)

    deltas = hist_probs - prob_estimates
    delta_bottom = np.percentile(deltas, percentile, axis=0)
    delta_top = np.percentile(deltas, 100 - percentile, axis=0)
    vals = (vals + 0.5)[:-1]
    return delta_bottom, delta_top, vals


if __name__ == '__main__':
    # bootstrap for calculating the confidence interval around the mean
    X = np.array([30, 37, 36, 43, 42, 43, 43, 46, 41, 42])
    X_bar = X.mean()
    bootstrapped_means = []
    for idx in bootstrap_idx(X.shape[0], n_bootstraps=100000):
        bootstrapped_means.append(X[idx].mean())

    bootstrapped_means = np.array(bootstrapped_means)
    deltas = X_bar - bootstrapped_means
    delta_1 = np.percentile(deltas, 10, axis=0)
    delta_9 = np.percentile(deltas, 90, axis=0)
    print((X_bar + delta_1, X_bar + delta_9))
