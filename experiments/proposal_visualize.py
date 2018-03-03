import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import torch
from rvi_sampling.distributions.proposal_distributions import SimonsProposal, SimonsSoftProposal, FunnelProposal
from rvi_sampling.utils.plotting import visualize_proposal, multi_quiver_plot
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser('Visualize Proposals')
    parser.add_argument('-nnp', '--nnpolicy', type=str, help='location of the neural net policy you want to visualize', default=False, required=False)
    parser.add_argument('-o', '--other', type=str, help='other proposal you want to visualize')
    args = parser.parse_args()

    if args.other == 'funnel':
        sp1 = FunnelProposal()
    elif args.other == 'soft':
        sp1 = SimonsSoftProposal()
    else:
        sp1 = SimonsProposal()

    if args.nnpolicy:
        sp2 = torch.load(args.nnpolicy)

to_plot = []
titles = []
t, x, x_arrows, y_arrows_normal = visualize_proposal([sp1], 50, 20, neural_network=False)
to_plot.append(y_arrows_normal)
titles.append('Hand Crafted')
# print(y_arrows_normal)
if args.nnpolicy:
    t, x, x_arrows, y_arrows_nn = visualize_proposal([sp2], 50, 20, neural_network=True)
    to_plot.append(y_arrows_nn)
    titles.append('Neural Network')

f = multi_quiver_plot(t, x, x_arrows,
                      to_plot,
                      titles,
                      figsize=(10,5))

f.savefig('visualized_proposals.pdf')
