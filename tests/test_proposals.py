import numpy as np
from rvi_sampling.distributions.proposal_distributions import FunnelProposal


def multi_draw(proposal, draw_args, n_draws=10000):
    # pool = multiprocessing.Pool()
    # pool.map(proposal)
    return np.mean([proposal.draw(*draw_args)[1] for _ in range(n_draws)])

def test_funnel_proposal():
    proposal = FunnelProposal(push_toward=[-1, 1])

    ####
    # trivial cases
    ####

    assert np.allclose(multi_draw(proposal, ([[5]], 2)), -1, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[-5]], 2)), +1, atol=0.05)

    ####
    # edge cases
    ####

    # there should never be a bias when you are at 0
    assert np.allclose(multi_draw(proposal, ([[0]], 0)), 0, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[0]], 1)), 0, atol=0.05)
    assert np.allclose(multi_draw(proposal, ([[0]], 2)), 0, atol=0.05)

    # if you're at +1, you could take a step +1 and go out of the window
    # therefore, this should be -1
    assert np.allclose(multi_draw(proposal, ([[+1]], 1)), -1, atol=0.05)
    # this should be zero, since in the next step you could be at x=2
    # with one time step, which means you can effectively correct yourself
    # and go back to 1
    assert np.allclose(multi_draw(proposal, ([[+1]], 2)), 0, atol=0.05)

    # this should be -1 because when you're at x=+3 and t=3
    # one wrong step could lead you to be out
    assert np.allclose(multi_draw(proposal, ([[+3]], 3)), -1, atol=0.05)

    # however, if you are at x=2 and t=3, this should be zero
    assert np.allclose(multi_draw(proposal, ([[+2]], 3)), 0, atol=0.05)

    # if you're at x=-1, at t=1 you could take a step -1 and go out of the window
    # therefore, if you have one step left this should be +1
    assert np.allclose(multi_draw(proposal, ([[-1]], 1)), 1, atol=0.05)

    # if you're at x=-1 at t=2, you could take a step -1 and end up at
    # x=-2 with t=1, since you still have one timestep left
    # you can effectively correct yourself. so this should be zero
    assert np.allclose(multi_draw(proposal, ([[-1]], 2)), 0, atol=0.05)

    # if you're at x=-2 at t=2, you could take a bad step -1 and end up
    # at x=-3 at t=1, however, you cant redeem yourself from there
    # therefore the bias should be +1
    assert np.allclose(multi_draw(proposal, ([[-2]], 2)), 1, atol=0.05)

    # if you're at x=-2 and t=3, you can take a "wrong step" to get to
    # x=-3 at t=2, however, you can still redeem yourself therefore bias = 0
    assert np.allclose(multi_draw(proposal, ([[-2]], 3)), 0, atol=0.05)

    # this should be +1 because when you're at x=-3 and t=3
    # one wrong step could lead you to be out
    assert np.allclose(multi_draw(proposal, ([[-3]], 3)), 1, atol=0.05)

