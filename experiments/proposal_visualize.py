import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import torch
from rvi_sampling.distributions.proposal_distributions import SimonsProposal, SimonsSoftProposal, FunnelProposal
from rvi_sampling.utils.plotting import visualize_proposal, multi_quiver_plot
import argparse
import seaborn as sns
sns.set_style('white')

WINDOW_SIZE = 5
INTERPOLATE_SIZE = 20
TIME = 30
PUSH_TOWARD = [-WINDOW_SIZE, WINDOW_SIZE]

if __name__=='__main__':
    parser = argparse.ArgumentParser('Visualize Proposals')
    parser.add_argument('-nnp', '--nnpolicy', type=str, help='location of the neural net policy you want to visualize', default=False, required=False)
    parser.add_argument('-o', '--other', type=str, help='other proposal you want to visualize')
    parser.add_argument('-t', '--title', type=str, help='Title for the plot')
    args = parser.parse_args()

    if args.other == 'funnel':
        sp1 = FunnelProposal(push_toward=PUSH_TOWARD)
    elif args.other == 'soft':
        sp1 = SimonsSoftProposal(push_toward=PUSH_TOWARD)
    else:
        sp1 = SimonsProposal()

    if args.nnpolicy:
        sp2 = torch.load(args.nnpolicy)

    to_plot = []
    titles = []
    if args.other:
        t, x, x_arrows, y_arrows_normal = visualize_proposal([sp1], TIME, INTERPOLATE_SIZE, neural_network=False)
        to_plot.append(y_arrows_normal)
        titles.append(args.title or 'Hand Crafted')

    # print(y_arrows_normal)
    if args.nnpolicy:
        t, x, x_arrows, y_arrows_nn = visualize_proposal([sp2], TIME, INTERPOLATE_SIZE, neural_network=True)
        to_plot.append(y_arrows_nn)
        titles.append(args.title or 'Neural Network')

    f = multi_quiver_plot(t, x, x_arrows,
                          to_plot,
                          titles,
                          figsize=(4,4))

    f.savefig('visualized_proposals.pdf')
