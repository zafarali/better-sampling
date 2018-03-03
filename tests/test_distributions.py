import numpy as np
from rvi_sampling.distributions.prior_distributions import DiscreteUniform


def test_discrete_uniform():
    du = DiscreteUniform(dimensions=1, start=-5, n_numbers=10)
    assert du.pdf(np.array([5])) != 0
    assert du.pdf(np.array([-5])) != 0
    assert du.pdf(np.array([6])) == 0
    assert du.pdf(np.array([-6])) == 0
    pass