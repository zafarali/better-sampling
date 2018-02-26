import numpy as np
from rvi_sampling.distributions.proposal_distributions import FunnelProposal
import multiprocessing


def multi_draw(proposal, draw_args, n_draws=10000):
    # pool = multiprocessing.Pool()
    # pool.map(proposal)
    return np.mean([proposal.draw(*draw_args)[1] for _ in range(n_draws)])

def test_funnel_proposal():
    proposal = FunnelProposal(push_toward=[-1, 1])

    """
    At time step 0, we expect biases to happen at any x > 1 or x < -1:
    """
    print(multi_draw(proposal, ([[0]], 0)))
    assert np.allclose(multi_draw(proposal, ([[0]], 0)), 0, atol=0.05)
    print(multi_draw(proposal, ([[1]], 0)))
    assert np.allclose(multi_draw(proposal, ([[1]], 0)), 0, atol=0.05)
    print(multi_draw(proposal, ([[-1]], 0)))
    assert np.allclose(multi_draw(proposal, ([[-1]], 0)), 0, atol=0.05)
    print(multi_draw(proposal, ([[2]], 0)))
    assert np.allclose(multi_draw(proposal, ([[2]], 0)), -1, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[-2]], 0)), 1, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[2]], 0)), -1, atol=0.05)

    """
    At time step 1, we expect biases to happen at any x > 2 or x < -2
    since time_left = 1, step_size=1 
    (means that we have 1 step available to get into the window)
    which is possible at x=2 (by taking a step -1) but not at x=3
    """
    assert np.allclose(multi_draw(proposal, ([[0]], 1)), 0, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[1]], 1)), 0, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[2]], 1)), 0, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[3]], 1)), -1, atol=0.05)

def test_funnel_proposal_2():
    """
    Test simons requirements for the funnel
    """

    proposal = FunnelProposal(push_toward=[-5, 5])
    assert np.allclose(multi_draw(proposal, ([[-5]], 0)), 0, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[-4]], 0)), 0, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[-6]], 0)), 1, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[-7]], 0)), 1, atol=0.05)


    assert np.allclose(multi_draw(proposal, ([[-5]], 1)), 0, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[-4]], 1)), 0, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[-6]], 1)), 0, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[-7]], 1)), 1, atol=0.05)


    assert np.allclose(multi_draw(proposal, ([[-5]], 2)), 0, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[-4]], 2)), 0, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[-6]], 2)), 0, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[-7]], 2)), 0, atol=0.05)

