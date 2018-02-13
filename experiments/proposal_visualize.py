import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import torch
from rvi_sampling.distributions.proposal_distributions import SimonsProposal
from rvi_sampling.plotting import visualize_proposal, multi_quiver_plot
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser('Visualize Proposals')
    parser.add_argument('-p', '--policy', type=str, help='location of the policy you want to visualize')
    args = parser.parse_args()

    sp1 = SimonsProposal()
    sp2 = torch.load(args.policy)

t, x, x_arrows, y_arrows_normal = visualize_proposal([sp1], 50, 10, neural_network=False)
t, x, x_arrows, y_arrows_nn = visualize_proposal([sp2], 50, 10, neural_network=True)

f = multi_quiver_plot(t, x, x_arrows,
                      [y_arrows_normal, y_arrows_nn],
                      titles=['Hand Crafted', 'Neural Network'],
                      figsize=(10,5))

f.savefig('visualized_proposals.pdf')
